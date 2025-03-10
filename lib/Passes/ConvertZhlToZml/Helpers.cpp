#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Params.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/Builder.h>
#include <zklang/Dialect/ZML/IR/OpInterfaces.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/Typing/Materialize.h>
#include <zklang/Passes/ConvertZhlToZml/Helpers.h>

using namespace mlir;
using namespace zhl;

namespace zml {

mlir::Operation *findCallee(mlir::StringRef name, mlir::ModuleOp root) {
  auto calleeComp = root.lookupSymbol<ComponentInterface>(name);
  if (calleeComp) {
    return calleeComp;
  }

  // Zhl Component ops don't declare their symbols in the symbol table
  for (auto zhlOp : root.getOps<zirgen::Zhl::ComponentOp>()) {
    if (zhlOp.getName() == name) {
      return zhlOp;
    }
  }
  return nullptr;
}

bool calleeIsBuiltin(mlir::Operation *op) {
  if (auto zmlOp = mlir::dyn_cast<ComponentInterface>(op)) {
    return zmlOp.getBuiltin();
  }
  return false;
}

#define DEBUG_TYPE "zml-create-slot"

FlatSymbolRefAttr createSlot(
    zhl::ComponentSlot *slot, OpBuilder &builder, ComponentInterface component, Location loc
) {
  mlir::SymbolTable st(component);

  auto desiredName = mlir::StringAttr::get(component.getContext(), slot->getSlotName());
  LLVM_DEBUG(
      llvm::dbgs() << "Creating a slot in " << component.getName()
                   << "\nDesired name: " << desiredName << "\n"
  );
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(&component.getRegion().front().front());
  auto type = materializeTypeBinding(builder.getContext(), slot->getBinding());
  auto fieldDef = builder.create<FieldDefOp>(loc, desiredName, TypeAttr::get(type));

  LLVM_DEBUG(llvm::dbgs() << "Field op: " << fieldDef << "\n");
  // Insert the FieldDefOp into the symbol table to make sure it has an unique name within the
  // component
  auto name = st.insert(fieldDef);
  LLVM_DEBUG(llvm::dbgs() << "Name created by the symbol table: " << name << "\n");
  slot->rename(name.getValue());
  return mlir::FlatSymbolRefAttr::get(component.getContext(), name);
}

#undef DEBUG_TYPE

TypeBinding unwrapArrayNTimes(const TypeBinding &type, size_t count) {
  if (count == 0) {
    return type;
  }
  assert(type.isArray());
  auto inner = type.getArrayElement([]() { return mlir::InFlightDiagnostic(); });
  assert(succeeded(inner));
  return unwrapArrayNTimes(*inner, count - 1);
}

Value storeAndLoadSlot(
    ComponentSlot &slot, Value value, FlatSymbolRefAttr slotName, Type slotType, Location loc,
    Type compType, OpBuilder &builder, Value self
) {
  storeSlot(slot, value, slotName, slotType, loc, compType, builder, self);
  return loadSlot(slot, value.getType(), slotName, slotType, loc, builder, self);
}

Value loadSlot(
    ComponentSlot &slot, Type valueType, FlatSymbolRefAttr slotName, Type slotType, Location loc,
    OpBuilder &builder, Value self
) {
  auto compSlotBinding = slot.getBinding();
  auto compSlotIVs = slot.collectIVs();

  if (compSlotIVs.empty()) {
    // Read the temporary back to a SSA value
    return builder.create<ReadFieldOp>(loc, slotType, self, slotName);
  } else {
    // Read the array back to a SSA value
    auto arrayDataBis = builder.create<ReadFieldOp>(loc, slotType, self, slotName);

    // Read the value we wrote into the array back to a SSA value
    return builder.create<ReadArrayOp>(loc, valueType, arrayDataBis, compSlotIVs);
  }
}

void storeSlot(
    ComponentSlot &slot, Value value, FlatSymbolRefAttr slotName, Type slotType, Location loc,
    Type compType, OpBuilder &builder, Value self
) {
  auto compSlotBinding = slot.getBinding();
  auto compSlotIVs = slot.collectIVs();

  if (compSlotIVs.empty()) {
    assert(slotType == value.getType() && "result of construction and slot type must be the same");
    // Write the construction in a temporary
    builder.create<WriteFieldOp>(loc, self, slotName, value);
  } else {
    auto unwrappedBinding = unwrapArrayNTimes(compSlotBinding, compSlotIVs.size());
    auto unwrappedType = materializeTypeBinding(builder.getContext(), unwrappedBinding);
    assert(
        unwrappedType == value.getType() &&
        "result of construction and slot inner array type must be the same"
    );
    // Read the array from the slot field
    auto arrayData = builder.create<ReadFieldOp>(loc, slotType, self, slotName);
    assert(arrayData);
    assert(value);
    // Write into the array the value
    builder.create<WriteArrayOp>(loc, arrayData, compSlotIVs, value, true);

    // Write the array back into the field
    builder.create<WriteFieldOp>(loc, self, slotName, arrayData);
  }
}

Value storeAndLoadArraySlot(
    Value value, FlatSymbolRefAttr slotName, Type slotType, Location loc, Type compType,
    OpBuilder &builder, mlir::ValueRange ivs, Value self
) {
  // Read the array from the slot field
  auto arrayData = builder.create<ReadFieldOp>(loc, slotType, self, slotName);
  // Write into the array the value
  builder.create<WriteArrayOp>(loc, arrayData, ivs, value, true);

  // Write the array back into the field
  builder.create<WriteFieldOp>(loc, self, slotName, arrayData);

  // Read the array back to a SSA value
  auto arrayDataBis = builder.create<ReadFieldOp>(loc, slotType, self, slotName);

  // Read the value we wrote into the array back to a SSA value
  return builder.create<ReadArrayOp>(loc, value.getType(), arrayDataBis, ivs);
}

void materializeFieldTypes(
    zhl::TypeBinding &binding, MLIRContext *ctx, SmallVectorImpl<Type> &types,
    SmallVectorImpl<StringRef> &fields
) {
  llvm::StringMap<Type> map;
  SmallVector<StringRef> fieldsToSort;
  fieldsToSort.reserve(binding.getMembers().size());

  for (auto &[memberName, memberBinding] : binding.getMembers()) {
    assert(memberBinding.has_value());
    auto memberType = materializeTypeBinding(ctx, *memberBinding);
    map[memberName] = memberType;
    fieldsToSort.push_back(memberName);
  }

  std::sort(fieldsToSort.begin(), fieldsToSort.end());
  for (auto field : fieldsToSort) {
    types.push_back(map[field]);
  }
  fields.insert(fields.end(), fieldsToSort.begin(), fieldsToSort.end());
}

void constructFieldReads(
    TypeRange types, ArrayRef<StringRef> fieldNames, SmallVectorImpl<Value> &results,
    OpBuilder &builder, Value self, Location loc, zhl::TypeBinding &binding
) {
  for (auto [type, field] : llvm::zip_equal(types, fieldNames)) {
    auto memberBinding = binding.getMembers().at(field);
    assert(memberBinding.has_value());
    ComponentSlot *slot = mlir::dyn_cast_if_present<ComponentSlot>(memberBinding->getSlot());
    assert(slot);
    auto slotName = FlatSymbolRefAttr::get(builder.getStringAttr(slot->getSlotName()));
    auto slotType = materializeTypeBinding(builder.getContext(), slot->getBinding());
    results.push_back(loadSlot(*slot, type, slotName, slotType, loc, builder, self));
  }
}

FailureOr<Value> constructPODComponent(
    Operation *op, zhl::TypeBinding &binding, OpBuilder &builder, Value self,
    llvm::function_ref<mlir::Value()> superTypeValueCb
) {
  auto ctor = CtorCallBuilder::Make(op, binding, builder, self);
  if (failed(ctor)) {
    return failure();
  }

  auto loc = binding.getLocation();

  auto superTypeValue = superTypeValueCb();
  SmallVector<Type, 1> argTypes({superTypeValue.getType()});
  SmallVector<StringRef, 1> fieldNames({"$super"});
  SmallVector<Value, 1> args({superTypeValue});
  auto toReserve = 1 + binding.getMembers().size();
  argTypes.reserve(toReserve);
  fieldNames.reserve(toReserve);
  args.reserve(toReserve);

  materializeFieldTypes(binding, builder.getContext(), argTypes, fieldNames);
  constructFieldReads(
      ArrayRef(argTypes).drop_front(), ArrayRef(fieldNames).drop_front(), args, builder, self, loc,
      binding
  );

  return ctor->build(builder, loc, args);
}

void createPODComponent(
    zhl::TypeBinding &binding, mlir::OpBuilder &builder, mlir::SymbolTable &st
) {
  SmallVector<Type> argTypes;
  SmallVector<StringRef> fieldNames;

  ComponentBuilder cb;
  auto loc = binding.getLocation();
  cb.name(binding.getName()).isClosure().location(loc);

  auto superType = materializeTypeBinding(builder.getContext(), binding.getSuperType());
  argTypes = {superType};
  fieldNames = {"$super"};
  argTypes.reserve(1 + binding.getMembers().size());
  fieldNames.reserve(1 + binding.getMembers().size());

  materializeFieldTypes(binding, builder.getContext(), argTypes, fieldNames);
  for (auto [memberName, memberType] : llvm::zip_equal(fieldNames, argTypes)) {
    cb.field(memberName, memberType);
  }

  if (!binding.getGenericParamNames().empty()) {
    cb.forceGeneric().typeParams(binding.getGenericParamNames());
  }
  cb.defer([&](ComponentOp compOp) {
    auto actualName = st.insert(compOp);
    binding.setName(actualName.getValue());

    // Now that we know the final name of the component we can configure the rest
    cb.fillBody(
        argTypes, {materializeTypeBinding(builder.getContext(), binding)},
        [&](mlir::ValueRange args, mlir::OpBuilder &B) {
      auto componentType = materializeTypeBinding(B.getContext(), binding);
      // Reference to self
      auto self = B.create<SelfOp>(loc, componentType);
      // Store the fields
      for (auto [fieldName, arg] : llvm::zip_equal(fieldNames, args)) {
        B.create<WriteFieldOp>(loc, self, fieldName, arg);
      }
      // Return self
      B.create<mlir::func::ReturnOp>(loc, mlir::ValueRange({self}));
    }
    );
  });
  auto compOp = cb.build(builder);
  assert(compOp);
}

namespace {

/// Creates symbols following the pattern 'prefix + counter'.
/// This class does not own the prefix.
class IncrementalSymbolizer {
public:
  IncrementalSymbolizer(MLIRContext *Ctx, Twine Prefix) : ctx(Ctx), prefix(Prefix), counter(0) {}

  SymbolRefAttr operator*() const {
    return SymbolRefAttr::get(StringAttr::get(ctx, prefix + "_" + Twine(counter)));
  }

  IncrementalSymbolizer &operator++() {
    ++counter;
    return *this;
  }

private:
  MLIRContext *ctx;
  Twine prefix;
  uint32_t counter;
};

} // namespace

static void populateAffineMapsToSymbolsMapImpl(
    Type Type, DenseMap<AffineMap, SymbolRefAttr> &Symbols, IncrementalSymbolizer &Symbolizer
) {
  auto Component = mlir::dyn_cast<ComponentType>(Type);
  if (!Component) {
    return;
  }
  for (auto Attr : Component.getParams()) {
    if (auto TypeAttr = mlir::dyn_cast<mlir::TypeAttr>(Attr)) {
      populateAffineMapsToSymbolsMapImpl(TypeAttr.getValue(), Symbols, Symbolizer);
    }
    if (auto AffineMapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(Attr)) {
      auto Map = AffineMapAttr.getValue();
      Symbols.insert({Map, *Symbolizer});
      ++Symbolizer;
    }
  }
}

void populateAffineMapsToSymbolsMap(
    ArrayRef<Type> Types, DenseMap<AffineMap, SymbolRefAttr> &Symbols, MLIRContext *Ctx
) {
  IncrementalSymbolizer Symbolizer(Ctx, "Aff$");
  for (auto Type : Types) {
    populateAffineMapsToSymbolsMapImpl(Type, Symbols, Symbolizer);
  }
}

mlir::FailureOr<CtorCallBuilder> CtorCallBuilder::Make(
    mlir::Operation *op, mlir::Value value, const zhl::ZIRTypeAnalysis &typeAnalysis,
    mlir::OpBuilder &builder, mlir::Value self
) {
  auto binding = typeAnalysis.getType(value);
  if (failed(binding)) {
    return op->emitError() << "failed to type check";
  }

  return Make(op, *binding, builder, self);
}

mlir::FailureOr<CtorCallBuilder> CtorCallBuilder::Make(
    mlir::Operation *op, const zhl::TypeBinding &binding, mlir::OpBuilder &builder, mlir::Value self
) {
  auto rootModule = op->getParentOfType<mlir::ModuleOp>();
  auto *calleeComp = findCallee(binding.getName(), rootModule);
  if (!calleeComp) {
    return op->emitError() << "could not find component with name " << binding.getName();
  }
  auto constructorType = materializeTypeBindingConstructor(builder, binding);

  return CtorCallBuilder(
      constructorType, binding, op->getParentOfType<ComponentInterface>(), self,
      calleeIsBuiltin(calleeComp)
  );
}

/// Given an attribute materializes it into a Value if it's either a SymbolRefAttr or an
/// IntegerAttr. Any other kind of Attribute is considered malformed IR and will abort.
static Value materializeParam(Attribute attr, OpBuilder &builder, Location loc) {
  if (auto symAttr = mlir::dyn_cast<SymbolRefAttr>(attr)) {
    auto param = builder.create<LoadValParamOp>(
        loc, ComponentType::Val(builder.getContext()), symAttr.getRootReference()
    );
    return builder.create<ValToIndexOp>(loc, param);
  }
  if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
    return builder.create<arith::ConstantOp>(
        loc, builder.getIndexType(),
        builder.getIntegerAttr(builder.getIndexType(), intAttr.getValue())
    );
  }
  assert(false && "Cannot materialize something that is not a symbol or a literal integer");
}

mlir::Value
CtorCallBuilder::build(mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args) {
  auto buildPrologue = [&](mlir::Value v) {
    builder.create<ConstrainCallOp>(loc, v, args);
    return v;
  };

  auto genericParams = compBinding.getGenericParamsMapping();
  auto declaredParamsCount = genericParams.sizeOfDeclared();
  auto compParams = mlir::cast<ComponentType>(ctorType.getResult(0)).getParams();
  auto liftedCompParams = compParams.drop_front(declaredParamsCount);

  // Allocate here the values we may generate
  SmallVector<SmallVector<Value>> mapOperandsMem(liftedCompParams.size());
  // And store a ValueRange pointing to the vector here
  SmallVector<ValueRange> mapOperands;
  // This idiom does not use any dimensions
  SmallVector<int32_t> dimsPerMap(liftedCompParams.size(), 0);
  mapOperands.reserve(liftedCompParams.size());

  size_t idx = 0;
  for (auto attr : liftedCompParams) {
    auto liftedConstExpr = mlir::cast<ConstExprAttr>(attr);
    auto &values = mapOperandsMem[idx];
    for (auto formal : liftedConstExpr.getFormals()) {
      assert(
          formal < (compParams.size() - liftedCompParams.size()) &&
          "Can only use as map operands declared parameters"
      );
      values.push_back(materializeParam(compParams[formal], builder, loc));
    }

    mapOperands.push_back(values);
    idx++;
  }

  auto ref = builder.create<ConstructorRefOp>(
      loc, compBinding.getName(), mapOperands, builder.getDenseI32ArrayAttr(dimsPerMap), ctorType,
      isBuiltin
  );
  auto call = builder.create<mlir::func::CallIndirectOp>(loc, ref, args);
  Value compValue = call.getResult(0);

  if (!compBinding.getSlot()) {
    return buildPrologue(compValue);
  }

  auto compSlot = mlir::cast<zhl::ComponentSlot>(compBinding.getSlot());
  auto compSlotBinding = compSlot->getBinding();

  // Create the field
  auto name = createSlot(compSlot, builder, callerComponentOp, loc);
  auto slotType = materializeTypeBinding(builder.getContext(), compSlotBinding);

  compValue = storeAndLoadSlot(
      *compSlot, compValue, name, slotType, loc, callerComponentOp.getType(), builder, self
  );

  return buildPrologue(compValue);
}

mlir::FunctionType CtorCallBuilder::getCtorType() const { return ctorType; }

const zhl::TypeBinding &CtorCallBuilder::getBinding() const { return compBinding; }

CtorCallBuilder::CtorCallBuilder(
    mlir::FunctionType type, const zhl::TypeBinding &binding, ComponentInterface caller,
    mlir::Value selfValue, bool builtin
)
    : ctorType(type), compBinding(binding), isBuiltin(builtin), callerComponentOp(caller),
      self(selfValue) {
  assert(callerComponentOp);
  assert(self);
}

mlir::FailureOr<Value> coerceToArray(TypedValue<ComponentType> v, OpBuilder &builder) {
  auto arraySuper = v.getType().getFirstMatchingSuperType([](Type t) {
    if (auto ct = mlir::dyn_cast_if_present<ComponentType>(t)) {
      return ct.isConcreteArray();
    }
    return false;
  });

  if (failed(arraySuper)) {
    return failure();
  }

  return builder.create<SuperCoerceOp>(v.getLoc(), *arraySuper, v).getResult();
}

} // namespace zml

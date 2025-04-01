#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Params.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/Builder.h>
#include <zklang/Dialect/ZML/IR/OpInterfaces.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/Typing/Materialize.h>
#include <zklang/Dialect/ZML/Utils/BackVariables.h>
#include <zklang/Dialect/ZML/Utils/Helpers.h>
#include <zklang/LanguageSupport/ZIR/BackVariables.h>

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

template <typename T, typename Fn> T queryComp(mlir::Operation *op, Fn query, T Default = T()) {
  if (auto zmlOp = mlir::dyn_cast<ComponentInterface>(op)) {
    return query(zmlOp);
  }
  return Default;
}

bool calleeIsBuiltin(mlir::Operation *op) {
  return queryComp<bool>(op, [](auto zmlOp) { return zmlOp.getBuiltin(); });
}

static bool calleeUsesBackVariablesHelper(mlir::Operation *op) {
  return queryComp<bool>(op, [](auto zmlOp) { return zmlOp.getUsesBackVariables(); }, true);
}

#define DEBUG_TYPE "zml-slot-helpers"

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
    OpBuilder &builder, Value self
) {
  storeSlot(slot, value, slotName, slotType, loc, builder, self);
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
    OpBuilder &builder, Value self
) {
  auto compSlotBinding = slot.getBinding();
  auto compSlotIVs = slot.collectIVs();

  if (compSlotIVs.empty()) {
    LLVM_DEBUG(
        llvm::dbgs() << "slotType == " << slotType << " | value.getType() == " << value.getType()
                     << "\n"
    );
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

#undef DEBUG_TYPE

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
    llvm::function_ref<mlir::Value()> superTypeValueCb, const zhl::TypeBindings &bindings
) {
  auto ctor = CtorCallBuilder::Make(op, binding, builder, self, bindings);
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

mlir::FailureOr<CtorCallBuilder> CtorCallBuilder::Make(
    mlir::Operation *op, mlir::Value value, const zhl::ZIRTypeAnalysis &typeAnalysis,
    mlir::OpBuilder &builder, mlir::Value self, const zhl::TypeBindings &bindings
) {
  auto binding = typeAnalysis.getType(value);
  if (failed(binding)) {
    return op->emitError() << "failed to type check";
  }

  return Make(op, *binding, builder, self, bindings);
}

mlir::FailureOr<CtorCallBuilder> CtorCallBuilder::Make(
    mlir::Operation *op, const zhl::TypeBinding &binding, mlir::OpBuilder &builder,
    mlir::Value self, const zhl::TypeBindings &bindings
) {
  auto rootModule = op->getParentOfType<mlir::ModuleOp>();
  auto *calleeComp = findCallee(binding.getName(), rootModule);
  if (!calleeComp) {
    return op->emitError() << "could not find component with name " << binding.getName();
  }
  auto constructorType = materializeTypeBindingConstructor(builder, binding, bindings);
  bool usesBackVariables = calleeUsesBackVariablesHelper(calleeComp);
  if (usesBackVariables) {
    ZML_BVDialectHelper TP;
    constructorType = lang::zir::injectBVFunctionParams(TP, constructorType);
  }

  return CtorCallBuilder(
      constructorType, binding, op->getParentOfType<ComponentInterface>(), self,
      calleeIsBuiltin(calleeComp), usesBackVariables
  );
}

Value CtorCallBuilder::buildCallWithoutSlot(OpBuilder &builder, Location loc, ValueRange args) {
  auto genericParams = compBinding.getGenericParamsMapping();
  auto ref = builder.create<ConstructorRefOp>(
      loc, compBinding.getName(), genericParams.size() - genericParams.sizeOfDeclared(), ctorType,
      isBuiltin
  );

  auto call = builder.create<mlir::func::CallIndirectOp>(loc, ref, args);
  builder.create<ConstrainCallOp>(loc, call.getResult(0), args);
  return call.getResult(0);
}

Value CtorCallBuilder::build(OpBuilder &builder, Location loc, ValueRange argsRange) {
  if (!compBinding.getSlot()) {
    return buildCallWithoutSlot(builder, loc, argsRange);
  }

  SmallVector<Value> args(argsRange);

  auto compSlot = mlir::cast<zhl::ComponentSlot>(compBinding.getSlot());
  // Generate the concrete type that is required based in the scope (i.e. inside an array frame this
  // will generate an array type wrapping the inner type).
  auto compSlotBinding = compSlot->getBinding();
  // Create the field
  auto name = createSlot(compSlot, builder, callerComponentOp, loc);
  auto slotType = materializeTypeBinding(builder.getContext(), compSlotBinding);

  if (usesBackVariables) {
    ZML_BVDialectHelper TP(*compSlot, builder.getContext());
    auto parentBV = lang::zir::loadBVValues(TP, callerComponentOp.getBodyFunc());
    auto BV =
        lang::zir::loadMemoryForField(parentBV, name, ctorType.getResult(0), TP, builder, loc);
    lang::zir::injectBVArgs(BV, args);
  }

  auto genericParams = compBinding.getGenericParamsMapping();
  auto ref = builder.create<ConstructorRefOp>(
      loc, compBinding.getName(), genericParams.size() - genericParams.sizeOfDeclared(), ctorType,
      isBuiltin
  );

  auto call = builder.create<mlir::func::CallIndirectOp>(loc, ref, args);
  Value compValue = call.getResult(0);

  compValue = storeAndLoadSlot(*compSlot, compValue, name, slotType, loc, builder, self);
  builder.create<ConstrainCallOp>(loc, compValue, args);
  return compValue;
}

mlir::FunctionType CtorCallBuilder::getCtorType() const { return ctorType; }

const zhl::TypeBinding &CtorCallBuilder::getBinding() const { return compBinding; }

CtorCallBuilder::CtorCallBuilder(
    mlir::FunctionType type, const zhl::TypeBinding &binding, ComponentInterface caller,
    mlir::Value selfValue, bool builtin, bool usesBVs
)
    : ctorType(type), compBinding(binding), isBuiltin(builtin), usesBackVariables(usesBVs),
      callerComponentOp(caller), self(selfValue) {
  assert(callerComponentOp);
  assert(self);
  assert(
      (!usesBackVariables || compBinding.getSlot()) &&
      "If the component uses back-variables it needs to have a slot!"
  );
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

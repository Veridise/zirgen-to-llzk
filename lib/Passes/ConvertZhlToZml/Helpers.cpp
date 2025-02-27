#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
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
  auto compSlotBinding = slot.getBinding();
  auto compSlotIVs = slot.collectIVs();
  storeSlot(slot, value, slotName, slotType, loc, compType, builder, self);

  if (compSlotIVs.empty()) {
    // Read the temporary back to a SSA value
    return builder.create<ReadFieldOp>(loc, slotType, self, slotName);
  } else {
    // Read the array back to a SSA value
    auto arrayDataBis = builder.create<ReadFieldOp>(loc, slotType, self, slotName);

    // Read the value we wrote into the array back to a SSA value
    return builder.create<ReadArrayOp>(loc, value.getType(), arrayDataBis, compSlotIVs);
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

mlir::Value
CtorCallBuilder::build(mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args) {
  auto buildPrologue = [&](mlir::Value v) {
    builder.create<ConstrainCallOp>(loc, v, args);
    return v;
  };

  auto ref = builder.create<ConstructorRefOp>(loc, ctorType, compBinding.getName(), isBuiltin);
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

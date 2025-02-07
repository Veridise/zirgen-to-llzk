#include "zklang/Passes/ConvertZhlToZml/Helpers.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include "zklang/Dialect/ZML/Typing/Materialize.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZML/IR/OpInterfaces.h>

using namespace mlir;
using namespace zkc::Zmir;
using namespace zhl;

namespace zkc {

// ComponentArity::ComponentArity() : isVariadic(false), paramCount(0) {}

// ComponentArity getComponentConstructorArity(zirgen::Zhl::ComponentOp op) {
//   ComponentArity arity;
//
//   // Add locations for each index and keep them sorted
//   std::map<uint32_t, mlir::Location> locsByIndex;
//   for (auto paramOp : op.getOps<zirgen::Zhl::ConstructorParamOp>()) {
//     arity.isVariadic = arity.isVariadic || paramOp.getVariadic();
//     arity.paramCount = std::max({arity.paramCount, paramOp.getIndex() + 1});
//     locsByIndex.insert({paramOp.getIndex(), paramOp.getLoc()});
//   }
//
//   // The iterator will be sorted since it's a `std::map`.
//   std::transform(
//       locsByIndex.begin(), locsByIndex.end(), std::back_inserter(arity.locs),
//       [](auto &pair) { return pair.second; }
//   );
//
//   return arity;
// }

// mlir::FlatSymbolRefAttr createTempField(
//     mlir::Location loc, mlir::Type type, mlir::OpBuilder &builder, Zmir::ComponentInterface op
// ) {
//   mlir::SymbolTable st(op);
//   auto desiredName = mlir::StringAttr::get(op.getContext(), "$temp");
//   mlir::OpBuilder::InsertionGuard guard(builder);
//   builder.setInsertionPointAfter(&op.getRegion().front().front());
//   auto fieldDef = builder.create<Zmir::FieldDefOp>(loc, desiredName, TypeAttr::get(type));
//   return mlir::FlatSymbolRefAttr::get(op.getContext(), st.insert(fieldDef));
// }
//
// /// Creates a temporary field to store the value and a sequence of reads and writes
// /// that disconnect the value creation from its users.
// mlir::Operation *storeValueInTemporary(
//     mlir::Location loc, Zmir::ComponentOp callerComp, mlir::Type fieldType, mlir::Value value,
//     mlir::ConversionPatternRewriter &rewriter
// ) {
//   // Create the field
//   auto name = createTempField(loc, fieldType, rewriter, callerComp);
//   // Write the construction in a temporary
//   auto self = rewriter.create<Zmir::SelfOp>(loc, callerComp.getType());
//   rewriter.create<Zmir::WriteFieldOp>(loc, self, name, value);
//
//   // Read the temporary back to a SSA value
//   return rewriter.create<Zmir::ReadFieldOp>(loc, fieldType, self, name);
// }

mlir::Operation *findCallee(mlir::StringRef name, mlir::ModuleOp root) {
  auto calleeComp = root.lookupSymbol<Zmir::ComponentInterface>(name);
  if (calleeComp) {
    return calleeComp;
  }

  // Zhl Component ops don't declare its symbols in the symbol table
  for (auto zhlOp : root.getOps<zirgen::Zhl::ComponentOp>()) {
    if (zhlOp.getName() == name) {
      return zhlOp;
    }
  }
  return nullptr;
}

bool calleeIsBuiltin(mlir::Operation *op) {
  if (auto zmlOp = mlir::dyn_cast<Zmir::ComponentInterface>(op)) {
    return zmlOp.getBuiltin();
  }
  return false;
}

FlatSymbolRefAttr createSlot(
    zhl::ComponentSlot *slot, OpBuilder &builder, Zmir::ComponentInterface component, Location loc
) {
  mlir::SymbolTable st(component);

  auto desiredName = mlir::StringAttr::get(component.getContext(), slot->getSlotName());
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(&component.getRegion().front().front());
  auto type = Zmir::materializeTypeBinding(builder.getContext(), slot->getBinding());
  auto fieldDef = builder.create<Zmir::FieldDefOp>(loc, desiredName, TypeAttr::get(type));

  // Insert the FieldDefOp into the symbol table to make sure it has an unique name within the
  // component
  return mlir::FlatSymbolRefAttr::get(component.getContext(), st.insert(fieldDef));
}

Value storeAndLoadSlot(
    Value value, FlatSymbolRefAttr slotName, Type slotType, Location loc, Type compType,
    OpBuilder &builder, Value self
) {
  // Write the construction in a temporary
  builder.create<Zmir::WriteFieldOp>(loc, self, slotName, value);

  // Read the temporary back to a SSA value
  return builder.create<Zmir::ReadFieldOp>(loc, slotType, self, slotName);
}

Value storeAndLoadArraySlot(
    Value value, FlatSymbolRefAttr slotName, Type slotType, Location loc, Type compType,
    OpBuilder &builder, mlir::ValueRange ivs, Value self
) {
  // Read the array from the slot field
  auto arrayData = builder.create<Zmir::ReadFieldOp>(loc, slotType, self, slotName);
  // Write into the array the value
  builder.create<Zmir::WriteArrayOp>(loc, arrayData, ivs, value, true);

  // Write the array back into the field
  builder.create<Zmir::WriteFieldOp>(loc, self, slotName, arrayData);

  // Read the array back to a SSA value
  auto arrayDataBis = builder.create<Zmir::ReadFieldOp>(loc, slotType, self, slotName);

  // Read the value we wrote into the array back to a SSA value
  return builder.create<Zmir::ReadArrayOp>(loc, value.getType(), arrayDataBis, ivs);
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
  auto constructorType = Zmir::materializeTypeBindingConstructor(builder, binding);

  return CtorCallBuilder(
      constructorType, binding, mlir::dyn_cast<Zmir::ComponentInterface>(calleeComp),
      op->getParentOfType<Zmir::ComponentInterface>(), self, calleeIsBuiltin(calleeComp)
  );
}

TypeBinding unwrapArrayNTimes(
    const TypeBinding &type, size_t count, std::function<mlir::InFlightDiagnostic()> emitError
) {
  if (count == 0) {
    return type;
  }
  assert(type.isArray());
  auto inner = type.getArrayElement(emitError);
  assert(succeeded(inner));
  return unwrapArrayNTimes(*inner, count - 1, emitError);
}

mlir::Value
CtorCallBuilder::build(mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args) {
  auto buildPrologue = [&](mlir::Value v) {
    builder.create<Zmir::ConstrainCallOp>(loc, v, args);
    return v;
  };

  auto ref =
      builder.create<Zmir::ConstructorRefOp>(loc, ctorType, compBinding.getName(), isBuiltin);
  auto call = builder.create<mlir::func::CallIndirectOp>(loc, ref, args);
  Value compValue = call.getResult(0);

  if (!compBinding.getSlot()) {
    return buildPrologue(compValue);
  }

  auto compSlot = mlir::cast<zhl::ComponentSlot>(compBinding.getSlot());
  auto compSlotBinding = compSlot->getBinding();
  auto compSlotIVs = compSlot->collectIVs();

  // Create the field
  auto name = createSlot(compSlot, builder, callerComponentOp, loc);

  auto slotType = Zmir::materializeTypeBinding(builder.getContext(), compSlotBinding);

  if (compSlotIVs.empty()) {
    assert(
        slotType == compValue.getType() && "result of construction and slot type must be the same"
    );
    compValue = storeAndLoadSlot(
        compValue, name, slotType, loc, callerComponentOp.getType(), builder, self
    );
  } else {
    auto unwrappedBinding = unwrapArrayNTimes(compSlotBinding, compSlotIVs.size(), [&]() {
      return callerComponentOp->emitError();
    });
    auto unwrappedType = Zmir::materializeTypeBinding(builder.getContext(), unwrappedBinding);
    assert(
        unwrappedType == compValue.getType() &&
        "result of construction and slot inner array type must be the same"
    );
    compValue = storeAndLoadArraySlot(
        compValue, name, slotType, loc, callerComponentOp.getType(), builder, compSlotIVs, self
    );
  }
  return buildPrologue(compValue);
}

mlir::FunctionType CtorCallBuilder::getCtorType() const { return ctorType; }

const zhl::TypeBinding &CtorCallBuilder::getBinding() const { return compBinding; }

Zmir::ComponentInterface CtorCallBuilder::getCalleeComp() const { return calleeComponentOp; }

CtorCallBuilder::CtorCallBuilder(
    mlir::FunctionType type, const zhl::TypeBinding &binding, Zmir::ComponentInterface callee,
    Zmir::ComponentInterface caller, mlir::Value selfValue, bool builtin
)
    : ctorType(type), compBinding(binding), isBuiltin(builtin), calleeComponentOp(callee),
      callerComponentOp(caller), self(selfValue) {
  assert(calleeComponentOp);
  assert(callerComponentOp);
  assert(self);
}

} // namespace zkc

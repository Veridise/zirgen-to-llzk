#include "zklang/Passes/ConvertZhlToZml/Helpers.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include "zklang/Dialect/ZML/Typing/Materialize.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZML/IR/OpInterfaces.h>

using namespace mlir;
using namespace zkc::Zmir;

namespace zkc {

ComponentArity::ComponentArity() : isVariadic(false), paramCount(0) {}

ComponentArity getComponentConstructorArity(zirgen::Zhl::ComponentOp op) {
  ComponentArity arity;

  // Add locations for each index and keep them sorted
  std::map<uint32_t, mlir::Location> locsByIndex;
  for (auto paramOp : op.getOps<zirgen::Zhl::ConstructorParamOp>()) {
    arity.isVariadic = arity.isVariadic || paramOp.getVariadic();
    arity.paramCount = std::max({arity.paramCount, paramOp.getIndex() + 1});
    locsByIndex.insert({paramOp.getIndex(), paramOp.getLoc()});
  }

  // The iterator will be sorted since it's a `std::map`.
  std::transform(
      locsByIndex.begin(), locsByIndex.end(), std::back_inserter(arity.locs),
      [](auto &pair) { return pair.second; }
  );

  return arity;
}

mlir::FlatSymbolRefAttr createTempField(
    mlir::Location loc, mlir::Type type, mlir::OpBuilder &builder, Zmir::ComponentInterface op
) {
  mlir::SymbolTable st(op);
  auto desiredName = mlir::StringAttr::get(op.getContext(), "$temp");
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(&op.getRegion().front().front());
  auto fieldDef = builder.create<Zmir::FieldDefOp>(loc, desiredName, TypeAttr::get(type));
  return mlir::FlatSymbolRefAttr::get(op.getContext(), st.insert(fieldDef));
}

/// Creates a temporary field to store the value and a sequence of reads and writes
/// that disconnect the value creation from its users.
mlir::Operation *storeValueInTemporary(
    mlir::Location loc, Zmir::ComponentOp callerComp, mlir::Type fieldType, mlir::Value value,
    mlir::ConversionPatternRewriter &rewriter
) {
  // Create the field
  auto name = createTempField(loc, fieldType, rewriter, callerComp);
  // Write the construction in a temporary
  auto self = rewriter.create<Zmir::GetSelfOp>(loc, callerComp.getType());
  rewriter.create<Zmir::WriteFieldOp>(loc, self, name, value);

  // Read the temporary back to a SSA value
  return rewriter.create<Zmir::ReadFieldOp>(loc, fieldType, self, name);
}

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
    OpBuilder &builder
) {
  // Write the construction in a temporary
  auto self = builder.create<Zmir::GetSelfOp>(loc, compType);
  builder.create<Zmir::WriteFieldOp>(loc, self, slotName, value);

  // Read the temporary back to a SSA value
  return builder.create<Zmir::ReadFieldOp>(loc, slotType, self, slotName);
}

mlir::FailureOr<CtorCallBuilder> CtorCallBuilder::Make(
    mlir::Operation *op, mlir::Value value, const zhl::ZIRTypeAnalysis &typeAnalysis,
    mlir::OpBuilder &builder
) {
  auto binding = typeAnalysis.getType(value);
  if (failed(binding)) {
    return op->emitError() << "failed to type check";
  }
  auto rootModule = op->getParentOfType<mlir::ModuleOp>();
  auto *calleeComp = findCallee(binding->getName(), rootModule);
  if (!calleeComp) {
    return op->emitError() << "could not find component with name " << binding->getName();
  }
  auto constructorType = Zmir::materializeTypeBindingConstructor(builder, *binding);

  return CtorCallBuilder(
      constructorType, *binding, mlir::dyn_cast<Zmir::ComponentInterface>(calleeComp),
      op->getParentOfType<Zmir::ComponentInterface>(), calleeIsBuiltin(calleeComp)
  );
}

mlir::Value
CtorCallBuilder::build(mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args) {
  auto ref =
      builder.create<Zmir::ConstructorRefOp>(loc, ctorType, compBinding.getName(), isBuiltin);
  auto call = builder.create<mlir::func::CallIndirectOp>(loc, ref, args);
  Value compValue = call.getResult(0);
  if (compBinding.getSlot()) {
    auto compSlot = mlir::dyn_cast<zhl::ComponentSlot>(compBinding.getSlot());
    assert(compSlot && "Cannot construct a component over a non-component slot");

    auto slotType = Zmir::materializeTypeBinding(builder.getContext(), compSlot->getBinding());
    assert(
        slotType == compValue.getType() && "result of construction and slot type must be the same"
    );
    // Create the field
    auto name = createSlot(compSlot, builder, callerComponentOp, loc);
    compValue =
        storeAndLoadSlot(compValue, name, slotType, loc, callerComponentOp.getType(), builder);
  }
  builder.create<Zmir::ConstrainCallOp>(loc, compValue, args);
  return compValue;
}

mlir::FunctionType CtorCallBuilder::getCtorType() const { return ctorType; }

const zhl::TypeBinding &CtorCallBuilder::getBinding() const { return compBinding; }

Zmir::ComponentInterface CtorCallBuilder::getCalleeComp() const { return calleeComponentOp; }

CtorCallBuilder::CtorCallBuilder(
    mlir::FunctionType type, const zhl::TypeBinding &binding, Zmir::ComponentInterface callee,
    Zmir::ComponentInterface caller, bool builtin
)
    : ctorType(type), compBinding(binding), isBuiltin(builtin), calleeComponentOp(callee),
      callerComponentOp(caller) {
  assert(callee);
  assert(caller);
}

} // namespace zkc

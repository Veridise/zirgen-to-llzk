#include "zklang/Passes/ConvertZhlToZml/Helpers.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include "zklang/Dialect/ZML/Typing/Materialize.h"

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

  return CtorCallBuilder(constructorType, *binding, calleeIsBuiltin(calleeComp));
}

mlir::Value
CtorCallBuilder::build(mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args) {
  auto ref =
      builder.create<Zmir::ConstructorRefOp>(loc, ctorType, compBinding.getName(), isBuiltin);
  auto call = builder.create<mlir::func::CallIndirectOp>(loc, ref, args);
  return call.getResult(0);
}

mlir::FunctionType CtorCallBuilder::getCtorType() const { return ctorType; }

CtorCallBuilder::CtorCallBuilder(
    mlir::FunctionType type, const zhl::TypeBinding &binding, bool builtin
)
    : ctorType(type), compBinding(binding), isBuiltin(builtin) {}

} // namespace zkc

#include "zklang/Passes/ConvertZhlToZml/Helpers.h"
#include "zklang/Dialect/ZML/IR/Ops.h"

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

} // namespace zkc

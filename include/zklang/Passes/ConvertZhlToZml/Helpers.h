#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include <cstdint>
#include <vector>

namespace zkc {

struct ComponentArity {
  /// The last argument of a ZIR component can be variadic.
  bool isVariadic;
  uint32_t paramCount;
  std::vector<mlir::Location> locs;

  ComponentArity();
};

/// Searches all Zhl::ConstructorParamOp in the component
/// and returns the largest index declared by the ops.
ComponentArity getComponentConstructorArity(zirgen::Zhl::ComponentOp);

mlir::FlatSymbolRefAttr createTempField(
    mlir::Location loc, mlir::Type type, mlir::OpBuilder &builder, Zmir::ComponentInterface op
);

/// Creates a temporary field to store the value and a sequence of reads and writes
/// that disconnect the value creation from its users.
mlir::Operation *storeValueInTemporary(
    mlir::Location loc, Zmir::ComponentOp callerComp, mlir::Type fieldType, mlir::Value value,
    mlir::ConversionPatternRewriter &rewriter
);

} // namespace zkc

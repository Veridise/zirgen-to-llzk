#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LogicalResult.h>

// #include <mlir/IR/Diagnostics.h>

// #include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.cpp.inc"

namespace zkc::Zmir {

mlir::LogicalResult
ReadFieldOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  // TODO
  return mlir::success();
}

mlir::LogicalResult
WriteFieldOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  // TODO
  return mlir::success();
}

mlir::LogicalResult GetGlobalOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location>,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::OpaqueProperties,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  // TODO
  inferredReturnTypes.push_back(ValType::get(ctx));
  return mlir::success();
}

} // namespace zkc::Zmir

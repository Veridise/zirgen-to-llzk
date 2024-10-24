#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
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

} // namespace zkc::Zmir

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
ConstructOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  // TODO
  // if (mlir::failed(getStructType().verifySymbol(symbolTable,
  // getOperation()))) {
  //   return mlir::failure();
  // }
  // FieldDefOp field = getFieldDefOp(symbolTable);
  // if (!field) {
  //   return emitOpError() << "undefined struct field: @" << getFieldName();
  // }
  // if (field.getType() != getResult().getType()) {
  //   return emitOpError() << "field read has wrong type; expected " <<
  //   field.getType() << ", got "
  //                        << getResult().getType();
  // }
  return mlir::success();
}

mlir::ParseResult BodyOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  auto buildFuncType =
      [](mlir::Builder &builder, mlir::ArrayRef<mlir::Type> argTypes,
         mlir::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void BodyOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

} // namespace zkc::Zmir

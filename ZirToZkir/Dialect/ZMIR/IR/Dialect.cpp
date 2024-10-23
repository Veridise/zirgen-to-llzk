#include <mlir/IR/DialectImplementation.h>

#include "ZirToZkir/Dialect/ZMIR/IR/Attrs.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"

// TableGen'd implementation files
#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.cpp.inc"

// Need a complete declaration of storage classes
#define GET_TYPEDEF_CLASSES
#include "ZirToZkir/Dialect/ZMIR/IR/Types.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "ZirToZkir/Dialect/ZMIR/IR/Attrs.cpp.inc"

// -----
// ZKIRDialect
// -----

auto zkc::Zmir::ZmirDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "ZirToZkir/Dialect/ZMIR/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "ZirToZkir/Dialect/ZMIR/IR/Types.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "ZirToZkir/Dialect/ZMIR/IR/Attrs.cpp.inc"
  >();
  // clang-format on
}

#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Attrs.h" // IWYU pragma: keep
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"   // IWYU pragma: keep
#include <mlir/IR/DialectImplementation.h>

// TableGen'd implementation files
#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.inc.cpp"

// Need a complete declaration of storage classes
#define GET_TYPEDEF_CLASSES
#include "ZirToZkir/Dialect/ZMIR/IR/Types.inc.cpp"
#define GET_ATTRDEF_CLASSES
#include "ZirToZkir/Dialect/ZMIR/IR/Attrs.inc.cpp"

auto zkc::Zmir::ZmirDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "ZirToZkir/Dialect/ZMIR/IR/Ops.inc.cpp"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "ZirToZkir/Dialect/ZMIR/IR/Types.inc.cpp"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "ZirToZkir/Dialect/ZMIR/IR/Attrs.inc.cpp"
  >();
  // clang-format on
}

#include "zklang/Dialect/ZML/IR/Dialect.h"
#include "zklang/Dialect/ZML/IR/Attrs.h" // IWYU pragma: keep
#include "zklang/Dialect/ZML/IR/Ops.h"   // IWYU pragma: keep
#include <mlir/IR/DialectImplementation.h>

// TableGen'd implementation files
#include "zklang/Dialect/ZML/IR/Dialect.cpp.inc"

// Need a complete declaration of storage classes
#define GET_TYPEDEF_CLASSES
#include "zklang/Dialect/ZML/IR/Types.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "zklang/Dialect/ZML/IR/Attrs.cpp.inc"

auto zml::ZMLDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "zklang/Dialect/ZML/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "zklang/Dialect/ZML/IR/Types.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "zklang/Dialect/ZML/IR/Attrs.cpp.inc"
  >();
  // clang-format on
}

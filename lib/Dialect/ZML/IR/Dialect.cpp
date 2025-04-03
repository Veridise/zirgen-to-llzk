//===- Dialect.cpp - ZML Dialect --------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZML/IR/Attrs.h> // IWYU pragma: keep
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/IR/Ops.h> // IWYU pragma: keep

// TableGen'd implementation files
#include <zklang/Dialect/ZML/IR/Dialect.cpp.inc>

// Need a complete declaration of storage classes
#define GET_TYPEDEF_CLASSES
#include <zklang/Dialect/ZML/IR/Types.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <zklang/Dialect/ZML/IR/Attrs.cpp.inc>

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

using namespace mlir;

namespace zml {

Operation *
ZMLDialect::materializeConstant(OpBuilder &builder, Attribute attr, Type type, Location loc) {
  if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
    if (isa<ComponentType>(type)) {
      return builder.create<LitValOp>(loc, ComponentType::Val(builder.getContext()), intAttr);
    }

    if (isa<IndexType>(type)) {
      return builder.create<arith::ConstantIndexOp>(loc, intAttr.getInt());
    }
  }
  if (auto arrayAttr = mlir::dyn_cast<DenseI64ArrayAttr>(attr)) {
    return builder.create<LitValArrayOp>(loc, type, arrayAttr);
  }

  return nullptr;
}

} // namespace zml

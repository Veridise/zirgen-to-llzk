//===- Conversion.cpp - ExtVal conversion -----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <mlir/IR/Builders.h>
#include <zklang/Dialect/ZML/ExtVal/Conversion.h>

namespace zml::extval {

//==----------------------------------------------------------------------------------==//
// ValueWrap
//==----------------------------------------------------------------------------------==//

BaseConverter::ValueWrap::ValueWrap(
    mlir::Value Val, mlir::OpBuilder &Builder, const TypeHelper &Helper
)
    : val(Val), builder(&Builder), helper(&Helper) {}

BaseConverter::ValueWrap::ValueWrap(
    uint64_t Val, mlir::OpBuilder &Builder, const TypeHelper &Helper
)
    : val(Helper.createLitOp(Val, Builder)), builder(&Builder), helper(&Helper) {}

BaseConverter::ValueWrap::operator mlir::Value() const { return val; }

BaseConverter::ValueWrap BaseConverter::ValueWrap::inv() {
  return ValueWrap(helper->createInvOp(val, *builder), *builder, *helper);
}

BaseConverter::ValueWrap BaseConverter::ValueWrap::operator+(const ValueWrap &other) {
  return ValueWrap(helper->createAddOp(val, other.val, *builder), *builder, *helper);
}

BaseConverter::ValueWrap BaseConverter::ValueWrap::operator-(const ValueWrap &other) {
  return ValueWrap(helper->createSubOp(val, other.val, *builder), *builder, *helper);
}

BaseConverter::ValueWrap BaseConverter::ValueWrap::operator-() {
  return ValueWrap(helper->createNegOp(val, *builder), *builder, *helper);
}

BaseConverter::ValueWrap BaseConverter::ValueWrap::operator*(const ValueWrap &other) {
  return ValueWrap(helper->createMulOp(val, other.val, *builder), *builder, *helper);
}

//==----------------------------------------------------------------------------------==//
// TypeHelper
//==----------------------------------------------------------------------------------==//

void BaseConverter::TypeHelper::assertIsValidRepr(mlir::Type t) const {
  assert(t == createArrayRepr(t.getContext()) && "Type t does not match the target polynomial");
}

void BaseConverter::TypeHelper::assertIsValidRepr(mlir::Value v) const {
  assertIsValidRepr(v.getType());
}

//==----------------------------------------------------------------------------------==//
// BaseConverter
//==----------------------------------------------------------------------------------==//

BaseConverter::BaseConverter(const TypeHelper &Helper) : helper(&Helper) {}

} // namespace zml::extval

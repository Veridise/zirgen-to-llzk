//===- LLZKTypeConverter.h - LLZK type conversion from ZML ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a type converter subclass that converts ZML types into
// LLZK types.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Transforms/DialectConversion.h>
#include <string_view>
#include <unordered_set>
#include <zklang/Dialect/ZML/ExtVal/Conversion.h>
#include <zklang/FiniteFields/Field.h>

namespace llzk {

class LLZKTypeConverter : public mlir::TypeConverter,
                          public zml::extval::BaseConverter::TypeHelper {
public:
  LLZKTypeConverter(const ff::FieldData &field);

  /// Returns the Type that represents the extended element at the lower level.
  virtual mlir::Type createArrayRepr(mlir::MLIRContext *) const override;
  /// Collects a set of values into a single value, usually via an array creation op.
  virtual mlir::Value
  collectValues(mlir::ValueRange, mlir::Location, mlir::OpBuilder &) const override;
  /// Reads each element of the array that represents the extended field element and wraps them in
  /// ValueWrap instances.
  virtual mlir::SmallVector<zml::extval::BaseConverter::ValueWrap>
  wrapArrayValues(mlir::Value v, mlir::OpBuilder &builder) const override;

  /// Creates an operation that represents addition between elements of the array representation
  virtual mlir::Value createAddOp(mlir::Value, mlir::Value, mlir::OpBuilder &) const override;
  /// Creates an operation that represents subtraction between elements of the array
  /// representation
  virtual mlir::Value createSubOp(mlir::Value, mlir::Value, mlir::OpBuilder &) const override;
  /// Creates an operation that represents multiplication between elements of the array
  /// representation
  virtual mlir::Value createMulOp(mlir::Value, mlir::Value, mlir::OpBuilder &) const override;
  /// Creates an operation that represents negation of elements of the array representation
  virtual mlir::Value createNegOp(mlir::Value, mlir::OpBuilder &) const override;
  /// Creates an operation that represents the inverse of elements of the array representation
  virtual mlir::Value createInvOp(mlir::Value, mlir::OpBuilder &) const override;
  /// Creates an operation that represents a literal value of the inner type of the array
  /// representation
  virtual mlir::Value createLitOp(uint64_t, mlir::OpBuilder &) const override;
  /// Creates an operation that returns a boolean value indicating if the input value is equal to
  /// 0 or not
  virtual mlir::Value createIszOp(mlir::Value, mlir::OpBuilder &) const override;
  /// Creates an assert operation
  virtual mlir::Operation *
  createAssertOp(mlir::Value, mlir::StringAttr, mlir::OpBuilder &) const override;
  /// Creates a logical and operation
  virtual mlir::Value createAndOp(mlir::Value, mlir::Value, mlir::OpBuilder &) const override;

private:
  const ff::FieldData field;
  std::unordered_set<std::string_view> feltEquivalentTypes, extValBuiltins;
};

} // namespace llzk

//===- ZhlOpConversionPattern.h ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern base class that is used for defining patterns that lower ZHL
// operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>

namespace zhl {

/// A modified version of OpConversionPattern that automatically extracts the TypeBinding attribute
/// of the operation and it's operand values. It's meant to only work with operations in the ZHL
/// dialect and these operations have to be annotated with the type binding beforehand.
template <typename SourceOp> class ZhlOpConversionPattern : public mlir::ConversionPattern {
public:
  using OpAdaptor = typename SourceOp::Adaptor;
  using Binding = zml::TypeBindingAttr;
  using BindingsAdaptor = typename SourceOp::template GenericAdaptor<mlir::ArrayRef<Binding>>;

  ZhlOpConversionPattern(mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : mlir::ConversionPattern(SourceOp::getOperationName(), benefit, context) {}
  ZhlOpConversionPattern(
      const mlir::TypeConverter &typeConverter, mlir::MLIRContext *context,
      mlir::PatternBenefit benefit = 1
  )
      : mlir::ConversionPattern(typeConverter, SourceOp::getOperationName(), benefit, context) {}

  /// Wrappers around the ConversionPattern methods that pass the derived op
  /// type.
  mlir::LogicalResult match(mlir::Operation *op) const final {
    auto binding = Binding::get(op);
    if (!binding) {
      return mlir::failure();
    }
    auto operandBindings = getOperandBindings(op);
    if (llvm::any_of(operandBindings, [](auto t) { return t == nullptr; })) {
      return mlir::failure();
    }
    auto sourceOp = mlir::cast<SourceOp>(op);
    return match(sourceOp, binding, BindingsAdaptor(operandBindings, sourceOp));
  }

  void rewrite(
      mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter
  ) const final {
    auto sourceOp = mlir::cast<SourceOp>(op);
    auto operandBindings = getOperandBindings(op);
    rewrite(
        sourceOp, Binding::get(op), OpAdaptor(operands, sourceOp),
        BindingsAdaptor(operandBindings, sourceOp), rewriter
    );
  }

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter
  ) const final {
    auto sourceOp = mlir::cast<SourceOp>(op);
    auto operandBindings = getOperandBindings(op);
    return matchAndRewrite(
        sourceOp, Binding::get(op), OpAdaptor(operands, sourceOp),
        BindingsAdaptor(operandBindings, sourceOp), rewriter
    );
  }

  /// Rewrite and Match methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual mlir::LogicalResult match(SourceOp, Binding, BindingsAdaptor) const {
    llvm_unreachable("must override match or matchAndRewrite");
  }

  virtual void
  rewrite(SourceOp, Binding, OpAdaptor, BindingsAdaptor, mlir::ConversionPatternRewriter &) const {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  }

  virtual mlir::LogicalResult matchAndRewrite(
      SourceOp op, Binding opBinding, OpAdaptor adaptor, BindingsAdaptor bindings,
      mlir::ConversionPatternRewriter &rewriter
  ) const {
    if (mlir::failed(match(op, opBinding, bindings))) {
      return mlir::failure();
    }
    rewrite(op, opBinding, adaptor, bindings, rewriter);
    return mlir::success();
  }

private:
  using mlir::ConversionPattern::matchAndRewrite;

  mlir::SmallVector<Binding> getOperandBindings(mlir::Operation *op) const {
    return llvm::map_to_vector(op->getOperands(), [](mlir::Value v) { return Binding::get(v); });
  }
};

} // namespace zhl

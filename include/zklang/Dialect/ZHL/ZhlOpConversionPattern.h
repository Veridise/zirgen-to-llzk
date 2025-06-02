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

template <typename Op>
mlir::FailureOr<typename Op::template GenericAdaptor<mlir::ArrayRef<zml::TypeBindingAttr>>>
bindingsAdaptor(Op op) {
  auto bindings = llvm::map_to_vector(op->getOperands(), [](mlir::Value v) {
    return zml::TypeBindingAttr::get(v);
  });
  if (llvm::any_of(bindings, [](auto t) { return t == nullptr; })) {
    return mlir::failure();
  }
  return typename Op::template GenericAdaptor<mlir::ArrayRef<zml::TypeBindingAttr>>(bindings, op);
}

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
    auto sourceOp = mlir::cast<SourceOp>(op);
    auto operandBindings = bindingsAdaptor(sourceOp);
    if (failed(operandBindings)) {
      return mlir::failure();
    }
    return match(sourceOp, binding, *operandBindings);
  }

  void rewrite(
      mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter
  ) const final {
    auto sourceOp = mlir::cast<SourceOp>(op);
    auto operandBindings = bindingsAdaptor(sourceOp);
    rewrite(sourceOp, Binding::get(op), OpAdaptor(operands, sourceOp), *operandBindings, rewriter);
  }

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter
  ) const final {
    auto sourceOp = mlir::cast<SourceOp>(op);
    auto operandBindings = bindingsAdaptor(sourceOp);
    return matchAndRewrite(
        sourceOp, Binding::get(op), OpAdaptor(operands, sourceOp), *operandBindings, rewriter
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
};

} // namespace zhl

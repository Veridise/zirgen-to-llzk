//===- Patterns.h - Reusable patterns of ZML ops ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes common patterns or templates thereof used by several
// passes grouped here together to avoid duplicating code.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zklang/Dialect/ZML/IR/Ops.h>

namespace zml {

/// A strategy for ReplaceSelfWith that replaces the op's result with
/// a Value coming from the N-th argument of the parent.
template <int ArgN, typename Parent = mlir::func::FuncOp> class Arg {
public:
  template <typename Op>
  static mlir::LogicalResult generate(
      Op op, mlir::ConversionPatternRewriter &, const mlir::TypeConverter *,
      mlir::SmallVector<mlir::Value, 1> &values
  ) {
    auto func = op->template getParentOfType<Parent>();
    if (func.getNumArguments() <= ArgN) {
      return mlir::failure();
    }
    values.push_back(func.getArgument(ArgN));
    return mlir::success();
  }
};

/// A strategy for ReplaceSelfWith that replaces the op with a new one.
/// This new op must return the same amount of values.
/// Known limitations: The new operation cannot have any operands.
template <typename Op> class NewOp {
public:
  template <typename InOp>
  static mlir::LogicalResult generate(
      InOp op, mlir::ConversionPatternRewriter &rewriter, const mlir::TypeConverter *typeConverter,
      mlir::SmallVector<mlir::Value, 1> &values
  ) {
    mlir::SmallVector<mlir::Type, 1> convertedTypes;
    if (mlir::failed(typeConverter->convertTypes(op->getResultTypes(), convertedTypes))) {
      return mlir::failure();
    }
    auto newOp = rewriter.create<Op>(op.getLoc(), convertedTypes, mlir::ValueRange());
    values.push_back(newOp);
    return mlir::success();
  }
};

/// A pattern for replacing ZML's SelfOp with another Value or Operation. The exact behavior of the
/// replacement is defined by the Strategy type. All ops defined inside the SelfOp's region are
/// hoisted out before removing the op.
template <typename Strategy> class ReplaceSelfWith : public mlir::OpConversionPattern<SelfOp> {
public:
  using mlir::OpConversionPattern<SelfOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      SelfOp op, typename mlir::OpConversionPattern<SelfOp>::OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter
  ) const override {
    mlir::SmallVector<mlir::Value, 1> selfReplacement;
    if (mlir::failed(Strategy::generate(op, rewriter, getTypeConverter(), selfReplacement))) {
      return mlir::failure();
    }
    rewriter.inlineBlockBefore(&op.getRegion().front(), op, selfReplacement);
    rewriter.replaceOp(op, selfReplacement);
    return mlir::success();
  }
};

} // namespace zml

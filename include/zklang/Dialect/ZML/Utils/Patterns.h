#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zklang/Dialect/ZML/IR/Ops.h>

/*
 * Common patterns or templates thereof used by several passes grouped here together
 * to avoid duplicating code.
 */

namespace zml {

/// A strategy for ReplaceSelfWith that replaces the op's result with
/// a Value coming from the N-th argument of the parent.
template <int ArgN, typename Parent = mlir::func::FuncOp> class Arg {
public:
  template <typename Op>
  mlir::FailureOr<mlir::ValueRange>
  generate(Op op, mlir::ConversionPatternRewriter &, const mlir::TypeConverter *) const {
    auto func = op->template getParentOfType<Parent>();
    return mlir::ValueRange(func.getArgument(ArgN));
  }
};

/// A strategy for ReplaceSelfWith that replaces the op with a new one.
/// This new op must return the same amount of values.
/// Known limitations: The new operation cannot have any operands.
template <typename Op> class NewOp {
public:
  template <typename InOp>
  mlir::FailureOr<mlir::ValueRange> generate(
      InOp op, mlir::ConversionPatternRewriter &rewriter, const mlir::TypeConverter *typeConverter
  ) const {
    mlir::SmallVector<mlir::Type, 1> convertedTypes;
    if (mlir::failed(typeConverter->convertTypes(op->getResultTypes(), convertedTypes))) {
      return mlir::failure();
    }
    return mlir::ValueRange(rewriter.create<Op>(op.getLoc(), convertedTypes, mlir::ValueRange()));
  }
};

/// A pattern for replacing ZML's SelfOp with another Value or Operation. The exact behavior of the
/// replacement is defined by the Strategy type. All ops defined inside the SelfOp's region are
/// hoisted out before removing the op.
template <typename Strategy> class ReplaceSelfWith : public mlir::OpConversionPattern<SelfOp> {
  static_assert(std::is_default_constructible_v<Strategy>);

public:
  template <typename... Args>
  ReplaceSelfWith(Args &&...args)
      : mlir::OpConversionPattern<SelfOp>(std::forward<Args>(args)...), strategy{} {}

  mlir::LogicalResult matchAndRewrite(
      SelfOp op, typename mlir::OpConversionPattern<SelfOp>::OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter
  ) const override {
    mlir::FailureOr<mlir::ValueRange> selfReplacement =
        strategy.generate(op, rewriter, getTypeConverter());
    if (mlir::failed(selfReplacement)) {
      return mlir::failure();
    }
    rewriter.inlineBlockBefore(&op.getRegion().front(), op, *selfReplacement);
    rewriter.replaceOp(op, *selfReplacement);
    return mlir::success();
  }

private:
  Strategy strategy;
};

} // namespace zml

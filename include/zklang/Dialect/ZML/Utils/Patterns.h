#pragma once

#include "zklang/Dialect/ZML/IR/Ops.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace zkc::Zmir {

template <int ArgN, typename Parent = mlir::func::FuncOp> class Arg {
public:
  template <typename Op>
  mlir::FailureOr<mlir::ValueRange>
  generate(Op op, mlir::ConversionPatternRewriter &, const mlir::TypeConverter *) const {
    auto func = op->template getParentOfType<Parent>();
    return mlir::ValueRange(func.getArgument(ArgN));
  }
};

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

} // namespace zkc::Zmir

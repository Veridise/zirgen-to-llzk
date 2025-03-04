#include "zklang/Dialect/ZML/ExtVal/BabyBear.h"
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <llzk/Dialect/LLZK/IR/Types.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zklang/Dialect/ZML/IR/Ops.h>

using namespace mlir;
using namespace zml;

namespace zml::extval::babybear {

Value Converter::lowerOp(
    ExtAddOp op, ExtAddOp::Adaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  // Addition is an element wise addition
  auto lhs = getTypeHelper().wrapArrayValues(adaptor.getLhs(), rewriter);
  auto rhs = getTypeHelper().wrapArrayValues(adaptor.getRhs(), rewriter);
  SmallVector<Value> sums;
  sums.reserve(lhs.size());
  std::transform(
      lhs.begin(), lhs.end(), rhs.begin(), std::back_inserter(sums),
      [](auto lhsVal, auto rhsVal) { return lhsVal + rhsVal; }
  );
  return getTypeHelper().collectValues(sums, op.getLoc(), rewriter);
}

Value Converter::lowerOp(
    ExtSubOp op, ExtSubOp::Adaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  // Subtraction is an element wise subtraction
  auto lhs = getTypeHelper().wrapArrayValues(adaptor.getLhs(), rewriter);
  auto rhs = getTypeHelper().wrapArrayValues(adaptor.getRhs(), rewriter);
  SmallVector<Value> subs;
  subs.reserve(lhs.size());
  std::transform(
      lhs.begin(), lhs.end(), rhs.begin(), std::back_inserter(subs),
      [](auto lhsVal, auto rhsVal) { return lhsVal - rhsVal; }
  );
  return getTypeHelper().collectValues(subs, op.getLoc(), rewriter);
}

// These two methods below are based on the equivalent operations found in
// third-party/zirgen/zirgen/components/fpext.cpp

Value Converter::lowerOp(
    ExtMulOp op, ExtMulOp::Adaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto a = getTypeHelper().wrapArrayValues(adaptor.getLhs(), rewriter);
  auto b = getTypeHelper().wrapArrayValues(adaptor.getRhs(), rewriter);
  auto NBETA = -ValueWrap(field.beta, rewriter, getTypeHelper());

  auto out_0 = a[0] * b[0] + NBETA * (a[1] * b[3] + a[2] * b[2] + a[3] * b[1]);
  auto out_1 = a[0] * b[1] + a[1] * b[0] + NBETA * (a[2] * b[3] + a[3] * b[2]);
  auto out_2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + NBETA * (a[3] * b[3]);
  auto out_3 = a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0];

  return getTypeHelper().collectValues(
      ValueRange({out_0, out_1, out_2, out_3}), op.getLoc(), rewriter
  );
}

Value Converter::lowerOp(
    ExtInvOp op, ExtInvOp::Adaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto a = getTypeHelper().wrapArrayValues(adaptor.getIn(), rewriter);
  auto BETA = ValueWrap(field.beta, rewriter, getTypeHelper());
  auto b0 = a[0] * a[0] + BETA * (a[1] * (a[3] + a[3]) - a[2] * a[2]);
  auto b2 = a[0] * (a[2] + a[2]) - a[1] * a[1] + BETA * (a[3] * a[3]);
  auto c = b0 * b0 + BETA * b2 * b2;
  auto ic = c.inv();
  b0 = b0 * ic;
  b2 = b2 * ic;

  ValueRange arr(
      {a[0] * b0 + BETA * a[2] * b2, -a[1] * b0 - BETA * a[3] * b2, -a[0] * b2 + a[2] * b0,
       a[1] * b2 - a[3] * b0}
  );
  return getTypeHelper().collectValues(arr, op.getLoc(), rewriter);
}

Value Converter::lowerOp(
    MakeExtOp op, MakeExtOp::Adaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  // Creating a ExtVal from a Val is creating an array and putting the Val's content into the
  // first element. Padding with zeroes
  auto zero = getTypeHelper().createLitOp(0, rewriter);
  return getTypeHelper().collectValues(
      ValueRange({adaptor.getIn(), zero, zero, zero}), op.getLoc(), rewriter
  );
}

} // namespace zml::extval::babybear

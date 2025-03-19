#include <zklang/Dialect/ZML/ExtVal/Conversion.h>
#include <zklang/Dialect/ZML/ExtVal/Patterns.h>

#include <numeric>

using namespace zml;
using namespace mlir;
using namespace zml::extval;

namespace {

template <typename Op> class ExtValOpConversionPattern : public OpConversionPattern<Op> {
public:
  ExtValOpConversionPattern(const TypeConverter &tc, MLIRContext *ctx, const BaseConverter &Conv)
      : OpConversionPattern<Op>(tc, ctx), conv(&Conv) {}

  using OpAdaptor = typename OpConversionPattern<Op>::OpAdaptor;

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto val = getConverter().lowerOp(op, adaptor, rewriter);
    rewriter.replaceOp(op, val);
    return success();
  }

private:
  const BaseConverter &getConverter() const { return *conv; }

  const BaseConverter *conv;
};

using ExtAddOpLowering = ExtValOpConversionPattern<ExtAddOp>;
using ExtSubOpLowering = ExtValOpConversionPattern<ExtSubOp>;
using ExtMulOpLowering = ExtValOpConversionPattern<ExtMulOp>;
using ExtInvOpLowering = ExtValOpConversionPattern<ExtInvOp>;
using MakeExtOpLowering = ExtValOpConversionPattern<MakeExtOp>;

class EqzExtOpLowering : public OpConversionPattern<EqzExtOp> {
public:
  EqzExtOpLowering(const TypeConverter &tc, MLIRContext *ctx, const BaseConverter &Conv)
      : OpConversionPattern<EqzExtOp>(tc, ctx), conv(&Conv) {}

  LogicalResult matchAndRewrite(EqzExtOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter)
      const override {
    auto &helper = getConverter().getTypeHelper();
    auto a = helper.wrapArrayValues(adaptor.getExtval(), rewriter);

    // I'm not completely sure what are the exact semantics of this builtin but from what I could
    // gather it asserts that the values of the extended element are zero.
    // What I don't know is if it also emits a constraint.

    SmallVector<Value> iszs;
    std::transform(a.begin(), a.end(), std::back_inserter(iszs), [&](Value v) -> Value {
      return helper.createIszOp(v, rewriter);
    });
    auto mult = std::reduce(iszs.begin() + 1, iszs.end(), *iszs.begin(), [&](auto lhs, auto rhs) {
      return helper.createAndOp(lhs, rhs, rewriter);
    });
    Twine msg = "failed assertion: extended field element is not equal to zero";
    rewriter.replaceOp(op, helper.createAssertOp(mult, rewriter.getStringAttr(msg), rewriter));

    return success();
  }

private:
  const BaseConverter &getConverter() const { return *conv; }

  const BaseConverter *conv;
};

} // namespace

namespace zml {

void populateExtValToLlzkConversionPatterns(
    RewritePatternSet &patterns, const TypeConverter &typeConverter, MLIRContext *ctx,
    const BaseConverter &extValConverter
) {
  patterns.add<
      ExtAddOpLowering, ExtSubOpLowering, ExtMulOpLowering, ExtInvOpLowering, MakeExtOpLowering,
      EqzExtOpLowering>(typeConverter, ctx, extValConverter);
}

} // namespace zml

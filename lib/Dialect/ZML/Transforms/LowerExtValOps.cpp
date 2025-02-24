#include <cassert>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/DialectConversion.h>
#include <tuple>
#include <unordered_set>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZML/BuiltIns/BuiltIns.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/Types.h>
#include <zklang/Dialect/ZML/Transforms/PassDetail.h>
#include <zklang/Dialect/ZML/Typing/ZMLTypeConverter.h>

using namespace mlir;
using namespace zml;

namespace {

/// A simple wrapper that implements operators =, -, * for readability
class ValueWrap {
  Value val;
  OpBuilder *builder;

public:
  ValueWrap(Value Val, OpBuilder &Builder) : val(Val), builder(&Builder) {}

  ValueWrap(uint64_t Val, OpBuilder &Builder)
      : val(Builder.create<LitValOp>(
            Builder.getUnknownLoc(), ComponentType::Val(Builder.getContext()), Val
        )),
        builder(&Builder) {}

  operator Value() const { return val; }

  ValueWrap inv() {
    return ValueWrap(builder->create<InvOp>(val.getLoc(), val.getType(), val), *builder);
  }

  ValueWrap operator+(const ValueWrap &other) {
    return ValueWrap(builder->create<AddOp>(val.getLoc(), val.getType(), val, other.val), *builder);
  }

  ValueWrap operator-(const ValueWrap &other) {
    return ValueWrap(builder->create<SubOp>(val.getLoc(), val.getType(), val, other.val), *builder);
  }

  ValueWrap operator-() {
    return ValueWrap(builder->create<NegOp>(val.getLoc(), val.getType(), val), *builder);
  }

  ValueWrap operator*(const ValueWrap &other) {
    return ValueWrap(builder->create<MulOp>(val.getLoc(), val.getType(), val, other.val), *builder);
  }
};

struct FieldData {
  virtual ~FieldData() = default;

  uint64_t degree, prime, beta;

  virtual Value
  lowerOp(ExtAddOp op, ExtAddOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const = 0;
  virtual Value
  lowerOp(ExtSubOp op, ExtSubOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const = 0;
  virtual Value
  lowerOp(ExtMulOp op, ExtMulOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const = 0;
  virtual Value
  lowerOp(ExtInvOp op, ExtInvOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const = 0;
  virtual Value
  lowerOp(MakeExtOp op, MakeExtOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const = 0;

  Type createArrayRepr(MLIRContext *ctx) const {
    return ComponentType::Array(ctx, ComponentType::Val(ctx), degree);
  }

  void assertIsValidRepr(Type t) const {
    assert(t == createArrayRepr(t.getContext()) && "Type t does not match the target polynomial");
  }

  void assertIsValidRepr(Value v) const { assertIsValidRepr(v.getType()); }

  SmallVector<ValueWrap> wrapArrayValues(Value v, OpBuilder &builder) const {
    assertIsValidRepr(v);
    SmallVector<ValueWrap> vec;
    vec.reserve(degree);
    for (uint64_t i = 0; i < degree; i++) {
      auto it = builder.create<index::ConstantOp>(v.getLoc(), i);
      auto elt = builder.create<ReadArrayOp>(
          v.getLoc(), ComponentType::Val(builder.getContext()), v, ValueRange({it})
      );
      vec.push_back(ValueWrap(elt, builder));
    }

    return vec;
  }

protected:
  FieldData(uint64_t prime_, uint64_t degree_, uint64_t beta_)
      : degree(degree_), prime(prime_), beta(beta_) {}
};

// Only BabyBear is supported for now
namespace babybear {

class Field : public FieldData {
public:
  Field() : FieldData(15 * (1 << 27) + 1, 4, 11) {}

  Value lowerOp(ExtAddOp op, ExtAddOp::Adaptor adaptor, ConversionPatternRewriter &rewriter)
      const override {
    // Addition is an element wise addition
    auto lhs = wrapArrayValues(adaptor.getLhs(), rewriter);
    auto rhs = wrapArrayValues(adaptor.getRhs(), rewriter);
    SmallVector<Value> sums;
    sums.reserve(lhs.size());
    std::transform(
        lhs.begin(), lhs.end(), rhs.begin(), std::back_inserter(sums),
        [](auto lhsVal, auto rhsVal) { return lhsVal + rhsVal; }
    );
    return rewriter.create<NewArrayOp>(op.getLoc(), createArrayRepr(op.getContext()), sums);
  }

  Value lowerOp(ExtSubOp op, ExtSubOp::Adaptor adaptor, ConversionPatternRewriter &rewriter)
      const override {
    // Subtraction is an element wise subtraction
    auto lhs = wrapArrayValues(adaptor.getLhs(), rewriter);
    auto rhs = wrapArrayValues(adaptor.getRhs(), rewriter);
    SmallVector<Value> sums;
    sums.reserve(lhs.size());
    std::transform(
        lhs.begin(), lhs.end(), rhs.begin(), std::back_inserter(sums),
        [](auto lhsVal, auto rhsVal) { return lhsVal - rhsVal; }
    );
    return rewriter.create<NewArrayOp>(op.getLoc(), createArrayRepr(op.getContext()), sums);
  }

  Value lowerOp(ExtMulOp op, ExtMulOp::Adaptor adaptor, ConversionPatternRewriter &rewriter)
      const override {
    auto a = wrapArrayValues(adaptor.getLhs(), rewriter);
    auto b = wrapArrayValues(adaptor.getRhs(), rewriter);
    auto NBETA = -ValueWrap(beta, rewriter);

    auto out_0 = a[0] * b[0] + NBETA * (a[1] * b[3] + a[2] * b[2] + a[3] * b[1]);
    auto out_1 = a[0] * b[1] + a[1] * b[0] + NBETA * (a[2] * b[3] + a[3] * b[2]);
    auto out_2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + NBETA * (a[3] * b[3]);
    auto out_3 = a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0];

    return rewriter.create<NewArrayOp>(
        op.getLoc(), createArrayRepr(op.getContext()), ValueRange({out_0, out_1, out_2, out_3})
    );
  }

  Value lowerOp(ExtInvOp op, ExtInvOp::Adaptor adaptor, ConversionPatternRewriter &rewriter)
      const override {
    auto a = wrapArrayValues(adaptor.getIn(), rewriter);
    auto BETA = ValueWrap(beta, rewriter);
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
    return rewriter.create<NewArrayOp>(op.getLoc(), createArrayRepr(op.getContext()), arr);
  }

  Value lowerOp(MakeExtOp op, MakeExtOp::Adaptor adaptor, ConversionPatternRewriter &rewriter)
      const override {
    // Creating a ExtVal from a Val is creating an array and putting the Val's content into the
    // first element. Padding with zeroes

    auto zero = rewriter.create<LitValOp>(
        rewriter.getUnknownLoc(), ComponentType::Val(rewriter.getContext()), 0
    );
    return rewriter.create<NewArrayOp>(
        op.getLoc(), createArrayRepr(op.getContext()),
        ValueRange({adaptor.getIn(), zero, zero, zero})
    );
  }
};

} // namespace babybear

class ExtValTypeConverter : public TypeConverter {
public:
  ExtValTypeConverter(const FieldData &field) {
    addConversion([](Type t) { return t; });

    addConversion([&](ComponentType t) -> std::optional<Type> {
      if (t == ComponentType::ExtVal(t.getContext())) {
        return field.createArrayRepr(t.getContext());
      }
      return std::nullopt;
    });
  }
};

template <typename Op> class ExtValOpConversionPattern : public OpConversionPattern<Op> {
public:
  ExtValOpConversionPattern(const TypeConverter &tc, MLIRContext *ctx, const FieldData &Field)
      : OpConversionPattern<Op>(tc, ctx), field(&Field) {}

  using OpAdaptor = typename OpConversionPattern<Op>::OpAdaptor;

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto val = getField().lowerOp(op, adaptor, rewriter);
    rewriter.replaceOp(op, val);
    return mlir::success();
  }

private:
  const FieldData &getField() const { return *field; }

  const FieldData *field;
};

using PatExtAdd = ExtValOpConversionPattern<ExtAddOp>;
using PatExtSub = ExtValOpConversionPattern<ExtSubOp>;
using PatExtMul = ExtValOpConversionPattern<ExtMulOp>;
using PatExtInv = ExtValOpConversionPattern<ExtInvOp>;
using PatMakeExt = ExtValOpConversionPattern<MakeExtOp>;

class LowerExtValOpsPass : public zml::LowerExtValOpsBase<LowerExtValOpsPass> {

  void runOnOperation() override {
    auto op = getOperation();

    babybear::Field BabyBear;
    ExtValTypeConverter typeConverter(BabyBear);
    mlir::MLIRContext *ctx = op->getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<PatExtAdd, PatExtSub, PatExtMul, PatExtInv, PatMakeExt>(
        typeConverter, ctx, BabyBear
    );

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<
        ZMLDialect, mlir::func::FuncDialect, mlir::index::IndexDialect, mlir::scf::SCFDialect,
        mlir::arith::ArithDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();
    target.addIllegalOp<ExtAddOp, ExtSubOp, ExtMulOp, ExtInvOp, MakeExtOp>();

    if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace zml {

std::unique_ptr<OperationPass<zml::ComponentOp>> createLowerExtValOpsPass() {
  return std::make_unique<LowerExtValOpsPass>();
}

} // namespace zml

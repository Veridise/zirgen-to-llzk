#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

namespace zkc::Zmir {

class UpdateScfForOpTypes : public mlir::OpConversionPattern<mlir::scf::ForOp> {
public:
  using OpConversionPattern<mlir::scf::ForOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class UpdateScfYieldOpTypes : public mlir::OpConversionPattern<mlir::scf::YieldOp> {
public:
  using OpConversionPattern<mlir::scf::YieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class UpdateScfExecuteRegionOpTypes : public mlir::OpConversionPattern<mlir::scf::ExecuteRegionOp> {
public:
  using OpConversionPattern<mlir::scf::ExecuteRegionOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ExecuteRegionOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

class UpdateScfIfOpTypes : public mlir::OpConversionPattern<mlir::scf::IfOp> {
public:
  using OpConversionPattern<mlir::scf::IfOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

/// Lowers literal Vals
class LitValOpLowering : public mlir::OpConversionPattern<Zmir::LitValOp> {
public:
  using OpConversionPattern<Zmir::LitValOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Zmir::LitValOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class ComponentLowering : public mlir::OpConversionPattern<SplitComponentOp> {
public:
  using OpConversionPattern<SplitComponentOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SplitComponentOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class FieldDefOpLowering : public mlir::OpConversionPattern<FieldDefOp> {
public:
  using OpConversionPattern<FieldDefOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(FieldDefOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class FuncOpLowering : public mlir::OpConversionPattern<mlir::func::FuncOp> {
public:
  using OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class ReturnOpLowering : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
public:
  using OpConversionPattern<mlir::func::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::ReturnOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

class CallOpLowering : public mlir::OpConversionPattern<mlir::func::CallOp> {

public:
  using OpConversionPattern<mlir::func::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class CallIndirectOpLoweringInCompute
    : public mlir::OpConversionPattern<mlir::func::CallIndirectOp> {

public:
  using OpConversionPattern<mlir::func::CallIndirectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallIndirectOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

class WriteFieldOpLowering : public mlir::OpConversionPattern<Zmir::WriteFieldOp> {

public:
  using OpConversionPattern<Zmir::WriteFieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Zmir::WriteFieldOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class RemoveConstructorRefOp : public mlir::OpConversionPattern<Zmir::ConstructorRefOp> {

public:
  using OpConversionPattern<Zmir::ConstructorRefOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Zmir::ConstructorRefOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

template <typename FromOp, typename ToOp>
class LowerArithBuiltIns : public mlir::OpConversionPattern<FromOp> {
public:
  using mlir::OpConversionPattern<FromOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      FromOp op, typename mlir::OpConversionPattern<FromOp>::OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter
  ) const override {
    rewriter.replaceOpWithNewOp<ToOp>(op, adaptor.getOperands());
    return mlir::success();
  }
};

using LowerBitAnd = LowerArithBuiltIns<Zmir::BitAndOp, llzk::AndFeltOp>;
using LowerAdd = LowerArithBuiltIns<Zmir::AddOp, llzk::AddFeltOp>;
using LowerSub = LowerArithBuiltIns<Zmir::SubOp, llzk::SubFeltOp>;
using LowerMul = LowerArithBuiltIns<Zmir::MulOp, llzk::MulFeltOp>;
using LowerMod = LowerArithBuiltIns<Zmir::ModOp, llzk::ModFeltOp>;
using LowerInv = LowerArithBuiltIns<Zmir::InvOp, llzk::InvFeltOp>;
using LowerNeg = LowerArithBuiltIns<Zmir::NegOp, llzk::NegFeltOp>;

class LowerIsz : public mlir::OpConversionPattern<IsZeroOp> {
public:
  using mlir::OpConversionPattern<IsZeroOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(IsZeroOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class ValToI1OpLowering : public mlir::OpConversionPattern<ValToI1Op> {
public:
  using mlir::OpConversionPattern<ValToI1Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ValToI1Op, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class AssertOpLowering : public mlir::OpConversionPattern<AssertOp> {
public:
  using mlir::OpConversionPattern<AssertOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AssertOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerReadFieldOp : public mlir::OpConversionPattern<ReadFieldOp> {
public:
  using mlir::OpConversionPattern<ReadFieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReadFieldOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerConstrainOp : public mlir::OpConversionPattern<ConstrainOp> {
public:
  using mlir::OpConversionPattern<ConstrainOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ConstrainOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerInRangeOp : public mlir::OpConversionPattern<InRangeOp> {
public:
  using mlir::OpConversionPattern<InRangeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(InRangeOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerNewArrayOp : public mlir::OpConversionPattern<NewArrayOp> {
public:
  using mlir::OpConversionPattern<NewArrayOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NewArrayOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerReadArrayOp : public mlir::OpConversionPattern<ReadArrayOp> {
public:
  using mlir::OpConversionPattern<ReadArrayOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReadArrayOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerAllocArrayOp : public mlir::OpConversionPattern<AllocArrayOp> {
public:
  using mlir::OpConversionPattern<AllocArrayOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AllocArrayOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerIndexToValOp : public mlir::OpConversionPattern<IndexToValOp> {
public:
  using mlir::OpConversionPattern<IndexToValOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(IndexToValOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerValToIndexOp : public mlir::OpConversionPattern<ValToIndexOp> {
public:
  using mlir::OpConversionPattern<ValToIndexOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ValToIndexOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerArrayLengthOp : public mlir::OpConversionPattern<GetArrayLenOp> {
public:
  using mlir::OpConversionPattern<GetArrayLenOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(GetArrayLenOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerWriteArrayOp : public mlir::OpConversionPattern<WriteArrayOp> {
public:
  using mlir::OpConversionPattern<WriteArrayOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(WriteArrayOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerConstrainCallOp : public mlir::OpConversionPattern<ConstrainCallOp> {
public:
  using mlir::OpConversionPattern<ConstrainCallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ConstrainCallOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerNopOp : public mlir::OpConversionPattern<NopOp> {
public:
  using mlir::OpConversionPattern<NopOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NopOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerSuperCoerceOp : public mlir::OpConversionPattern<SuperCoerceOp> {
public:
  using mlir::OpConversionPattern<SuperCoerceOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SuperCoerceOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class LowerLoadValParamOp : public mlir::OpConversionPattern<LoadValParamOp> {
public:
  using mlir::OpConversionPattern<LoadValParamOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(LoadValParamOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

} // namespace zkc::Zmir

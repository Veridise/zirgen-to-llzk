#pragma once

#include "Helpers.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zkir/Dialect/ZKIR/IR/Ops.h"
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

namespace zkc::Zmir {

/// Lowers literal Vals
class LitValOpLowering : public mlir::OpConversionPattern<Zmir::LitValOp> {
public:
  using OpConversionPattern<Zmir::LitValOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Zmir::LitValOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

/// Converts self into a struct allocation
class GetSelfOpLowering : public mlir::OpConversionPattern<Zmir::GetSelfOp> {
public:
  using OpConversionPattern<Zmir::GetSelfOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Zmir::GetSelfOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class ComponentLowering : public mlir::OpConversionPattern<SplitComponentOp> {
public:
  using OpConversionPattern<SplitComponentOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SplitComponentOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class FieldDefOpLowering : public mlir::OpConversionPattern<FieldDefOp> {
public:
  using OpConversionPattern<FieldDefOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(FieldDefOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class FuncOpLowering : public mlir::OpConversionPattern<mlir::func::FuncOp> {
public:
  using OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class ReturnOpLowering
    : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
public:
  using OpConversionPattern<mlir::func::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::ReturnOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CallOpLowering : public mlir::OpConversionPattern<mlir::func::CallOp> {

public:
  using OpConversionPattern<mlir::func::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CallIndirectOpLoweringInCompute
    : public mlir::OpConversionPattern<mlir::func::CallIndirectOp> {

public:
  using OpConversionPattern<mlir::func::CallIndirectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallIndirectOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CallIndirectOpLoweringInConstrain
    : public mlir::OpConversionPattern<mlir::func::CallIndirectOp> {

public:
  using OpConversionPattern<mlir::func::CallIndirectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallIndirectOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class WriteFieldOpLowering
    : public mlir::OpConversionPattern<Zmir::WriteFieldOp> {

public:
  using OpConversionPattern<Zmir::WriteFieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Zmir::WriteFieldOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class RemoveConstructorRefOp
    : public mlir::OpConversionPattern<Zmir::ConstructorRefOp> {

public:
  using OpConversionPattern<Zmir::ConstructorRefOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Zmir::ConstructorRefOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

template <typename FromOp, typename ToOp>
class LowerArithBuiltIns : public mlir::OpConversionPattern<FromOp> {
public:
  using mlir::OpConversionPattern<FromOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(FromOp op,
                  typename mlir::OpConversionPattern<FromOp>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ToOp>(op, adaptor.getOperands());
    return mlir::success();
  }
};

using LowerBitAnd = LowerArithBuiltIns<Zmir::BitAndOp, zkir::AndFeltOp>;
using LowerAdd = LowerArithBuiltIns<Zmir::AddOp, zkir::AddFeltOp>;
using LowerSub = LowerArithBuiltIns<Zmir::SubOp, zkir::SubFeltOp>;
using LowerMul = LowerArithBuiltIns<Zmir::MulOp, zkir::MulFeltOp>;
using LowerInv = LowerArithBuiltIns<Zmir::InvOp, zkir::InvFeltOp>;
// TODO LowerIsz
using LowerNeg = LowerArithBuiltIns<Zmir::NegOp, zkir::NegFeltOp>;

class LowerReadFieldOp : public mlir::OpConversionPattern<ReadFieldOp> {
public:
  using mlir::OpConversionPattern<ReadFieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReadFieldOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class LowerConstrainOp : public mlir::OpConversionPattern<ConstrainOp> {
public:
  using mlir::OpConversionPattern<ConstrainOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ConstrainOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class LowerInRangeOp : public mlir::OpConversionPattern<InRangeOp> {
public:
  using mlir::OpConversionPattern<InRangeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(InRangeOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

} // namespace zkc::Zmir

#pragma once

#include "Helpers.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
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

} // namespace zkc::Zmir

#pragma once

#include <cstdint>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zklang/Dialect/ZML/ExtVal/Conversion.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/FiniteFields/BabyBear.h>

namespace zml::extval {

namespace babybear {

class Converter : public BaseConverter {
public:
  using BaseConverter::BaseConverter;

  mlir::Value lowerOp(
      zml::ExtAddOp op, zml::ExtAddOp::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;

  mlir::Value lowerOp(
      zml::ExtSubOp op, zml::ExtSubOp::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;

  mlir::Value lowerOp(
      zml::ExtMulOp op, zml::ExtMulOp::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;

  mlir::Value lowerOp(
      zml::ExtInvOp op, zml::ExtInvOp::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;

  mlir::Value lowerOp(
      zml::MakeExtOp op, zml::MakeExtOp::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;

private:
  ff::babybear::Field field;
};

} // namespace babybear

} // namespace zml::extval

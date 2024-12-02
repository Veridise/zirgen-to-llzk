#pragma once

#include "Helpers.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

namespace zkc {

/// Lowers literal Vals
class ZhlLiteralLowering
    : public mlir::OpConversionPattern<zirgen::Zhl::LiteralOp> {
public:
  using OpConversionPattern<zirgen::Zhl::LiteralOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::LiteralOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

/// Lowers literal Vals
class ZhlLiteralStrLowering
    : public mlir::OpConversionPattern<zirgen::Zhl::StringOp> {
public:
  using OpConversionPattern<zirgen::Zhl::StringOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::StringOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class FoldUnrealizedCasts
    : public mlir::OpConversionPattern<mlir::UnrealizedConversionCastOp> {

public:
  using OpConversionPattern<
      mlir::UnrealizedConversionCastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::UnrealizedConversionCastOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

/// Converts `zhl.parameter` op uses to the corresponding argument of the body
/// and updates the type of the argument.
class ZhlParameterLowering
    : public mlir::OpConversionPattern<zirgen::Zhl::ConstructorParamOp> {
public:
  using OpConversionPattern<
      zirgen::Zhl::ConstructorParamOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::ConstructorParamOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

/// Converts `zhl.construct` ops into calls to the body function of
/// a component.
class ZhlConstructLowering
    : public mlir::OpConversionPattern<zirgen::Zhl::ConstructOp> {
public:
  using OpConversionPattern<zirgen::Zhl::ConstructOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::ConstructOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;

private:
  void prepareArguments(mlir::ValueRange, mlir::ArrayRef<mlir::Type>,
                        mlir::Location, mlir::ConversionPatternRewriter &,
                        std::vector<mlir::Value> &) const;

  mlir::Value prepareArgument(mlir::Value, mlir::Type, mlir::Location,
                              mlir::ConversionPatternRewriter &) const;

  mlir::FailureOr<mlir::Type> getTypeFromName(mlir::StringRef) const;
};

/// Converts `zhl.constrain` ops to `zmir.constrain` ones.
class ZhlConstrainLowering
    : public mlir::OpConversionPattern<zirgen::Zhl::ConstraintOp> {
public:
  using OpConversionPattern<zirgen::Zhl::ConstraintOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::ConstraintOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

/// Removes `zhl.global` ops
class ZhlGlobalRemoval
    : public mlir::OpConversionPattern<zirgen::Zhl::GlobalOp> {
public:
  using OpConversionPattern<zirgen::Zhl::GlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::GlobalOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

/// Removes `zhl.declare` ops
class ZhlDeclarationRemoval
    : public mlir::OpConversionPattern<zirgen::Zhl::DeclarationOp> {
public:
  using OpConversionPattern<zirgen::Zhl::DeclarationOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::DeclarationOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

/// Lowers `zhl.define` ops to ZMIR component fields
class ZhlDefineLowering
    : public mlir::OpConversionPattern<zirgen::Zhl::DefinitionOp> {
public:
  using OpConversionPattern<zirgen::Zhl::DefinitionOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::DefinitionOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

/// Lowers `zhl.super` to ZMIR field operations
class ZhlSuperLoweringInFunc
    : public mlir::OpConversionPattern<zirgen::Zhl::SuperOp> {
public:
  using OpConversionPattern<zirgen::Zhl::SuperOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::SuperOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class ZhlSuperLoweringInMap
    : public mlir::OpConversionPattern<zirgen::Zhl::SuperOp> {
public:
  using OpConversionPattern<zirgen::Zhl::SuperOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::SuperOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class ZhlSuperLoweringInBlock
    : public mlir::OpConversionPattern<zirgen::Zhl::SuperOp> {
public:
  using OpConversionPattern<zirgen::Zhl::SuperOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::SuperOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

/// Lowers `zhl.extern` to a function declaration
class ZhlExternLowering
    : public mlir::OpConversionPattern<zirgen::Zhl::ExternOp> {
public:
  using OpConversionPattern<zirgen::Zhl::ExternOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::ExternOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

/// Lowers `zhl.lookup` to ZMIR component field read
class ZhlLookupLowering
    : public mlir::OpConversionPattern<zirgen::Zhl::LookupOp> {
public:
  using OpConversionPattern<zirgen::Zhl::LookupOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::LookupOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

/// Lowers `zhl.subscript` to ZMIR array read
class ZhlSubscriptLowering
    : public mlir::OpConversionPattern<zirgen::Zhl::SubscriptOp> {
public:
  using OpConversionPattern<zirgen::Zhl::SubscriptOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::SubscriptOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

/// Lowers `zhl.array` into a `zmir.new_array`
class ZhlArrayLowering
    : public mlir::OpConversionPattern<zirgen::Zhl::ArrayOp> {

public:
  using OpConversionPattern<zirgen::Zhl::ArrayOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::ArrayOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class ZhlCompToZmirCompPattern
    : public mlir::OpConversionPattern<zirgen::Zhl::ComponentOp> {
public:
  using OpConversionPattern<zirgen::Zhl::ComponentOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::ComponentOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class ZhlRangeOpLowering
    : public mlir::OpConversionPattern<zirgen::Zhl::RangeOp> {
public:
  using OpConversionPattern<zirgen::Zhl::RangeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::RangeOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class ZhlMapLowering : public mlir::OpConversionPattern<zirgen::Zhl::MapOp> {
public:
  using OpConversionPattern<zirgen::Zhl::MapOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::MapOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class ZhlBlockLowering
    : public mlir::OpConversionPattern<zirgen::Zhl::BlockOp> {
public:
  using OpConversionPattern<zirgen::Zhl::BlockOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::BlockOp, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

} // namespace zkc

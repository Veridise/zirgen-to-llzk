#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Dialect/ZHL/Typing/Analysis.h"
#include "zklang/Dialect/ZHL/Typing/TypeBindings.h"
#include "zklang/Passes/ConvertZhlToZml/Helpers.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/Typing/Materialize.h>

namespace zkc {

template <typename Op> class ZhlOpLoweringPattern : public mlir::OpConversionPattern<Op> {
public:
  template <typename... Args>
  ZhlOpLoweringPattern(zhl::ZIRTypeAnalysis &typeAnalysis, Args &&...args)
      : mlir::OpConversionPattern<Op>(std::forward<Args>(args)...), typeAnalysis(&typeAnalysis) {}

  mlir::FailureOr<zhl::TypeBinding> getType(mlir::Operation *op) const {
    return typeAnalysis->getType(op);
  }

  mlir::FailureOr<zhl::TypeBinding> getType(Op op) const {
    return typeAnalysis->getType(op.getOperation());
  }

  mlir::FailureOr<zhl::TypeBinding> getType(mlir::Value value) const {
    return typeAnalysis->getType(value);
  }

  mlir::FailureOr<zhl::TypeBinding> getType(mlir::StringRef name) const {
    return typeAnalysis->getType(name);
  }

  /// Extracts the binding from the input value/operation and creates a
  /// cast of the value into the type materialized from the binding.
  /// If the super type is specified generates an additional SuperCoerceOp
  /// from the binding's type to the super type.
  /// REQUIRES that the operation has one result only.
  mlir::FailureOr<mlir::Value> getCastedValue(
      mlir::Operation *op, mlir::OpBuilder &builder,
      mlir::SmallVector<mlir::Operation *, 2> &generatedOps, mlir::Type super = nullptr
  ) const {
    assert(
        op->getNumResults() == 1 && "casting helper can only work with operations with 1 result"
    );
    return getCastedValue(op->getResult(0), builder, super);
  }

  mlir::FailureOr<mlir::Value>
  getCastedValue(mlir::Operation *op, mlir::OpBuilder &builder, mlir::Type super = nullptr) const {
    mlir::SmallVector<mlir::Operation *, 2> genOps;
    return getCastedValue(op, builder, genOps, super);
  }

  mlir::FailureOr<mlir::Value> getCastedValue(
      Op op, mlir::OpBuilder &builder, mlir::SmallVector<mlir::Operation *, 2> &generatedOps,
      mlir::Type super = nullptr
  ) const {
    return getCastedValue(op.getOperation(), builder, super);
  }

  mlir::FailureOr<mlir::Value>
  getCastedValue(Op op, mlir::OpBuilder &builder, mlir::Type super = nullptr) const {
    mlir::SmallVector<mlir::Operation *, 2> generatedOps;
    return getCastedValue(op.getOperation(), builder, generatedOps, super);
  }

  mlir::FailureOr<mlir::Value> getCastedValue(
      mlir::Value value, mlir::OpBuilder &builder,
      mlir::SmallVector<mlir::Operation *, 2> &generatedOps, mlir::Type super = nullptr
  ) const {
    auto binding = getType(value);
    if (mlir::failed(binding)) {
      return mlir::failure();
    }
    return getCastedValue(value, *binding, builder, generatedOps, super);
  }

  mlir::FailureOr<mlir::Value>
  getCastedValue(mlir::Value value, mlir::OpBuilder &builder, mlir::Type super = nullptr) const {
    auto binding = getType(value);
    if (mlir::failed(binding)) {
      return mlir::failure();
    }
    mlir::SmallVector<mlir::Operation *, 2> generatedOps;
    return getCastedValue(value, *binding, builder, generatedOps, super);
  }

  /// A non-failing version that takes a binding as additional parameter
  mlir::Value getCastedValue(
      mlir::Value value, const zhl::TypeBinding &binding, mlir::OpBuilder &builder,
      mlir::SmallVector<mlir::Operation *, 2> &generatedOps, mlir::Type super = nullptr
  ) const {
    auto materialized = Zmir::materializeTypeBinding(builder.getContext(), binding);
    assert(materialized);
    if (value.getType() == materialized) {
      return value;
    }
    auto cast = builder.create<mlir::UnrealizedConversionCastOp>(
        value.getLoc(), mlir::TypeRange(materialized), mlir::ValueRange(value)
    );
    generatedOps.push_back(cast.getOperation());

    mlir::Value result = cast.getResult(0);
    if (super && super != materialized) {
      auto coerce = builder.create<Zmir::SuperCoerceOp>(value.getLoc(), super, result);
      generatedOps.push_back(coerce.getOperation());
      result = coerce;
    }
    return result;
  }

  mlir::Value getCastedValue(
      mlir::Value value, const zhl::TypeBinding &binding, mlir::OpBuilder &builder,
      mlir::Type super = nullptr
  ) const {
    mlir::SmallVector<mlir::Operation *, 2> generatedOps;
    return getCastedValue(value, binding, builder, generatedOps, super);
  }

  mlir::FailureOr<CtorCallBuilder>
  makeCtorCallBuilder(mlir::Operation *op, mlir::Value value, mlir::OpBuilder &builder) const {
    auto selfOp = op->getParentOfType<Zmir::SelfOp>();
    if (!selfOp) {
      return op->emitOpError() << "is not within a self region";
    }
    return CtorCallBuilder::Make(op, value, *typeAnalysis, builder, selfOp.getSelfValue());
  }

  const zhl::TypeBindings &getTypeBindings() const { return typeAnalysis->getBindings(); }

private:
  const zhl::ZIRTypeAnalysis *typeAnalysis;
};

/// Lowers literal Vals
class ZhlLiteralLowering : public ZhlOpLoweringPattern<zirgen::Zhl::LiteralOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::LiteralOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::LiteralOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Lowers literal Strings
class ZhlLiteralStrLowering : public ZhlOpLoweringPattern<zirgen::Zhl::StringOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::StringOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::StringOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Converts `zhl.parameter` op uses to the corresponding argument of the body
/// and updates the type of the argument.
class ZhlParameterLowering : public ZhlOpLoweringPattern<zirgen::Zhl::ConstructorParamOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::ConstructorParamOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::ConstructorParamOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Converts `zhl.construct` ops into calls to the body function of
/// a component.
class ZhlConstructLowering : public ZhlOpLoweringPattern<zirgen::Zhl::ConstructOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::ConstructOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::ConstructOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;

private:
  void
  prepareArguments(mlir::ValueRange, mlir::ArrayRef<mlir::Type>, mlir::Location, mlir::ConversionPatternRewriter &, std::vector<mlir::Value> &)
      const;

  mlir::Value
  prepareArgument(mlir::Value, mlir::Type, mlir::Location, mlir::ConversionPatternRewriter &) const;
};

/// Converts `zhl.constrain` ops to `zmir.constrain` ones.
class ZhlConstrainLowering : public ZhlOpLoweringPattern<zirgen::Zhl::ConstraintOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::ConstraintOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::ConstraintOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Removes `zhl.global` ops
class ZhlGlobalRemoval : public ZhlOpLoweringPattern<zirgen::Zhl::GlobalOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::GlobalOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::GlobalOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Remove compiler directive ops
class ZhlDirectiveRemoval : public ZhlOpLoweringPattern<zirgen::Zhl::DirectiveOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::DirectiveOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::DirectiveOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Remove `zhl.generic` ops
class ZhlGenericRemoval : public ZhlOpLoweringPattern<zirgen::Zhl::TypeParamOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::TypeParamOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::TypeParamOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Remove `zhl.specialize` ops
class ZhlSpecializeRemoval : public ZhlOpLoweringPattern<zirgen::Zhl::SpecializeOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::SpecializeOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::SpecializeOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Removes `zhl.declare` ops
class ZhlDeclarationRemoval : public ZhlOpLoweringPattern<zirgen::Zhl::DeclarationOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::DeclarationOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::DeclarationOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Lowers `zhl.define` ops to ZMIR component fields
class ZhlDefineLowering : public ZhlOpLoweringPattern<zirgen::Zhl::DefinitionOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::DefinitionOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::DefinitionOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Lowers `zhl.super` to ZMIR field operations
class ZhlSuperLoweringInFunc : public ZhlOpLoweringPattern<zirgen::Zhl::SuperOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::SuperOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::SuperOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

class ZhlSuperLoweringInMap : public ZhlOpLoweringPattern<zirgen::Zhl::SuperOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::SuperOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::SuperOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

class ZhlSuperLoweringInBlock : public ZhlOpLoweringPattern<zirgen::Zhl::SuperOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::SuperOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::SuperOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

class ZhlSuperLoweringInSwitch : public ZhlOpLoweringPattern<zirgen::Zhl::SuperOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::SuperOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::SuperOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Lowers `zhl.extern` to a function declaration
class ZhlExternLowering : public ZhlOpLoweringPattern<zirgen::Zhl::ExternOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::ExternOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::ExternOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Lowers `zhl.lookup` to ZMIR component field read
class ZhlLookupLowering : public ZhlOpLoweringPattern<zirgen::Zhl::LookupOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::LookupOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::LookupOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Lowers `zhl.subscript` to ZMIR array read
class ZhlSubscriptLowering : public ZhlOpLoweringPattern<zirgen::Zhl::SubscriptOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::SubscriptOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::SubscriptOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

/// Lowers `zhl.array` into a `zmir.new_array`
class ZhlArrayLowering : public ZhlOpLoweringPattern<zirgen::Zhl::ArrayOp> {

public:
  using ZhlOpLoweringPattern<zirgen::Zhl::ArrayOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::ArrayOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

class ZhlCompToZmirCompPattern : public ZhlOpLoweringPattern<zirgen::Zhl::ComponentOp> {
public:
  /*using ZhlOpLoweringPattern<zirgen::Zhl::ComponentOp>::ZhlOpLoweringPattern;*/

  template <typename... Args>
  ZhlCompToZmirCompPattern(std::function<void(mlir::StringRef)> delegate, Args &&...args)
      : ZhlOpLoweringPattern<zirgen::Zhl::ComponentOp>(std::forward<Args>(args)...),
        builtinOverriden(delegate) {}

  mlir::LogicalResult matchAndRewrite(
      zirgen::Zhl::ComponentOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  ) const override;

private:
  std::function<void(mlir::StringRef)> builtinOverriden;
};

class ZhlRangeOpLowering : public ZhlOpLoweringPattern<zirgen::Zhl::RangeOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::RangeOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::RangeOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

class ZhlMapLowering : public ZhlOpLoweringPattern<zirgen::Zhl::MapOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::MapOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::MapOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;
};

class ZhlBlockLowering : public ZhlOpLoweringPattern<zirgen::Zhl::BlockOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::BlockOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::BlockOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

class ZhlReduceLowering : public ZhlOpLoweringPattern<zirgen::Zhl::ReduceOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::ReduceOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::ReduceOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

class ZhlSwitchLowering : public ZhlOpLoweringPattern<zirgen::Zhl::SwitchOp> {
public:
  using ZhlOpLoweringPattern<zirgen::Zhl::SwitchOp>::ZhlOpLoweringPattern;

  mlir::LogicalResult
  matchAndRewrite(zirgen::Zhl::SwitchOp, OpAdaptor, mlir::ConversionPatternRewriter &)
      const override;
};

} // namespace zkc

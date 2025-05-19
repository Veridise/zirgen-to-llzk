//===- Patterns.cpp - ZHL->LLZK struct conversion patterns ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the available conversion patterns for converting ZHL
// component operations into LLZK struct and function operations.
//
//===----------------------------------------------------------------------===//

#include <zklang/Passes/ConvertZhlToLlzkFelt/Patterns.h>

#include <cstdint>
#include <limits>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <llzk/Dialect/Felt/IR/Dialect.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Function/IR/Dialect.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/LLZK/IR/AttributeHelper.h>
#include <llzk/Dialect/Struct/IR/Dialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/LLZK/Builder.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/TypeInterfaces.h>

using namespace zklang;
using namespace mlir;
using namespace zirgen::Zhl;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk::felt;

static zml::FixedTypeBindingAttr getTypeBinding(Operation *op, StringRef name = "zml.binding") {
  return op->getAttrOfType<zml::FixedTypeBindingAttr>(name);
}

static zml::FixedTypeBindingAttr getTypeBinding(Value val) {
  return TypeSwitch<Value, zml::FixedTypeBindingAttr>(val)
      .Case([](OpResult res) { return getTypeBinding(res.getDefiningOp()); })
      .Case([](BlockArgument arg) {
    SmallString<30> name("zml.arg_binding.");
    Twine(arg.getArgNumber()).toVector(name);
    return getTypeBinding(arg.getParentBlock()->getParentOp(), name);
  }).Default([](auto) { return nullptr; });
}

namespace {

//===----------------------------------------------------------------------===//
// LowerLiteralToLlzkFeltOp
//===----------------------------------------------------------------------===//

class LowerLiteralToLlzkFeltOp : public OpConversionPattern<LiteralOp> {
public:
  using OpConversionPattern<LiteralOp>::OpConversionPattern;

  LogicalResult match(LiteralOp op) const override {
    return success(getTypeBinding(op.getOperation()));
  }

  void rewrite(LiteralOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto *tc = getTypeConverter();
    assert(tc);
    auto value = op.getValue();
    assert(value <= std::numeric_limits<int64_t>::max());
    rewriter.replaceOpWithNewOp<llzk::felt::FeltConstantOp>(
        op, llzk::felt::FeltConstAttr::get(
                getContext(), llzk::toAPInt(static_cast<int64_t>(op.getValue()))
            )
    );
  }
};

//===----------------------------------------------------------------------===//
// LowerConstructToLlzkFeltOp
//===----------------------------------------------------------------------===//

static Value unwrapToFelt(ConversionPatternRewriter *rewriter, Value arg) {
  auto Val = mlir::cast<zml::ComponentLike>(FeltType::get(arg.getContext()));
  if (mlir::isa<FeltType>(arg.getType())) {
    return arg;
  }
  if (auto comp = mlir::dyn_cast<zml::ComponentLike>(arg.getType())) {
    if (comp.subtypeOf(Val)) {
      return rewriter->create<zml::SuperCoerceOp>(arg.getLoc(), Val, arg);
    }
  }
  if (mlir::isa<ExprType>(arg.getType())) {
    auto cast = rewriter->create<UnrealizedConversionCastOp>(
        arg.getLoc(), getTypeBinding(arg).getType(), arg
    );
    return unwrapToFelt(rewriter, cast.getResult(0));
  }
  llvm_unreachable("unhandled value case");
  return nullptr;
}

template <typename Replacement>
class LowerConstructToLlzkFeltOp : public OpConversionPattern<ConstructOp> {
public:
  LowerConstructToLlzkFeltOp(StringRef Name, const TypeConverter &tc, MLIRContext *ctx)
      : OpConversionPattern<ConstructOp>(tc, ctx), name(Name) {}

  LogicalResult match(ConstructOp op) const override {
    auto typeBinding = getTypeBinding(op.getOperation());

    return success(typeBinding && typeBinding.getName() == name && typeBinding.getBuiltin());
  }

  void
  rewrite(ConstructOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> args =
        llvm::map_to_vector(adaptor.getArgs(), std::bind_front(unwrapToFelt, &rewriter));

    rewriter.replaceOpWithNewOp<Replacement>(
        op, TypeRange({FeltType::get(getContext())}), ValueRange(args), ArrayRef<NamedAttribute>()
    );

    // TODO: Memory allocations!!
  }

private:
  StringRef name;
};

} // namespace

void zklang::populateZhlToLlzkFeltConversionPatterns(
    const TypeConverter &tc, RewritePatternSet &patterns
) {
  auto *ctx = patterns.getContext();
  patterns
      // clang-format off
      .add<LowerLiteralToLlzkFeltOp>(tc, ctx)
      .add<LowerConstructToLlzkFeltOp<AndFeltOp>>("BitAnd", tc, ctx)
      .add<LowerConstructToLlzkFeltOp<AddFeltOp>>("Add", tc, ctx)
      .add<LowerConstructToLlzkFeltOp<SubFeltOp>>("Sub", tc, ctx)
      .add<LowerConstructToLlzkFeltOp<MulFeltOp>>("Mul", tc, ctx)
      .add<LowerConstructToLlzkFeltOp<ModFeltOp>>("Mod", tc, ctx)
      .add<LowerConstructToLlzkFeltOp<InvFeltOp>>("Inv", tc, ctx)
      .add<LowerConstructToLlzkFeltOp<NegFeltOp>>("Neg", tc, ctx)
      // clang-format on
      ;
}

void zklang::populateZhlToLlzkFeltConversionTarget(ConversionTarget &target) {
  target.addLegalDialect<ZhlDialect, StructDialect, FeltDialect, FunctionDialect, zml::ZMLDialect>(
  );
  target.addIllegalOp<LiteralOp>();
  target.addLegalDialect<BuiltinDialect>();
}

void zklang::populateZhlToLlzkFeltConversionPatternsAndLegality(
    const TypeConverter &tc, RewritePatternSet &patterns, ConversionTarget &target
) {
  populateZhlToLlzkFeltConversionPatterns(tc, patterns);
  populateZhlToLlzkFeltConversionTarget(target);
}

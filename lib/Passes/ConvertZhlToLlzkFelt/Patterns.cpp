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

#include <optional>
#include <zklang/Passes/ConvertZhlToLlzkFelt/Patterns.h>

#include <cstdint>
#include <limits>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/ErrorHandling.h>
#include <llzk/Dialect/Felt/IR/Dialect.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Types.h>
#include <llzk/Dialect/Function/IR/Dialect.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/LLZK/IR/AttributeHelper.h>
#include <llzk/Dialect/Struct/IR/Dialect.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/LLZK/Builder.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>
#include <zklang/Dialect/ZHL/ZhlOpConversionPattern.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/IR/OpInterfaces.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/TypeInterfaces.h>
#include <zklang/Dialect/ZML/Typing/Materialize.h>
#include <zklang/Dialect/ZML/Utils/Helpers.h>

using namespace zklang;
using namespace mlir;
using namespace zirgen::Zhl;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk::felt;
using namespace zhl;

static bool isTargetConstructOp(ConstructOp op, StringSet<> names) {
  auto typeBinding = zml::TypeBindingAttr::get(op.getOperation());

  return typeBinding && names.contains(typeBinding->getName()) && typeBinding->isBuiltin();
}

namespace {

//===----------------------------------------------------------------------===//
// LowerLiteralToLlzkFeltOp
//===----------------------------------------------------------------------===//

class LowerLiteralToLlzkFeltOp : public ZhlOpConversionPattern<LiteralOp> {
public:
  using ZhlOpConversionPattern<LiteralOp>::ZhlOpConversionPattern;

  LogicalResult match(LiteralOp, Binding, BindingsAdaptor) const final { return success(); }

  void rewrite(
      LiteralOp op, Binding, OpAdaptor, BindingsAdaptor, ConversionPatternRewriter &rewriter
  ) const final {
    auto value = op.getValue();
    assert(value <= std::numeric_limits<int64_t>::max());
    rewriter.replaceOpWithNewOp<llzk::felt::FeltConstantOp>(
        op, llzk::felt::FeltConstAttr::get(getContext(), llzk::toAPInt(static_cast<int64_t>(value)))
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
        arg.getLoc(), zml::TypeBindingAttr::get(arg).materializeType(), arg
    );
    return unwrapToFelt(rewriter, cast.getResult(0));
  }
  llvm_unreachable("unhandled value case");
  return nullptr;
}

/// Base pattern for lowering construct ops to felt native operations.
class LowerConstructToLlzkFeltOpBase : public ZhlOpConversionPattern<ConstructOp> {
public:
  LowerConstructToLlzkFeltOpBase(StringRef Name, const TypeConverter &tc, MLIRContext *ctx)
      : ZhlOpConversionPattern<ConstructOp>(tc, ctx), name(Name) {}

  LogicalResult match(ConstructOp op, Binding, BindingsAdaptor) const final {
    bool isTargetFunc = true;
    auto scope = scopeFunction();
    if (scope.has_value()) {
      isTargetFunc = zml::opIsInFunc(*scope, op);
    }
    return success(isTargetConstructOp(op, {name}) && isTargetFunc);
  }

  virtual std::optional<StringRef> scopeFunction() const { return std::nullopt; }

private:
  StringRef name;
};

/// Pattern for operations that can exist in both compute and constrain.
template <typename Replacement>
class LowerConstructToLlzkFeltOp : public LowerConstructToLlzkFeltOpBase {
public:
  using LowerConstructToLlzkFeltOpBase::LowerConstructToLlzkFeltOpBase;

  void rewrite(
      ConstructOp op, Binding, OpAdaptor adaptor, BindingsAdaptor,
      ConversionPatternRewriter &rewriter
  ) const final {
    SmallVector<Value> args =
        llvm::map_to_vector(adaptor.getArgs(), std::bind_front(unwrapToFelt, &rewriter));

    rewriter.replaceOpWithNewOp<Replacement>(
        op, TypeRange({FeltType::get(getContext())}), ValueRange(args), ArrayRef<NamedAttribute>()
    );
  }
};

/// Pattern for ops that can only run in the witness generator that are located in the compute
/// function. Lowers to the native operation plus operations for storing the result in a
/// field.
template <typename Replacement>
class LowerConstructToLlzkWitnessGenFeltOpInCompute : public LowerConstructToLlzkFeltOpBase {
public:
  using LowerConstructToLlzkFeltOpBase::LowerConstructToLlzkFeltOpBase;

  std::optional<StringRef> scopeFunction() const final { return "compute"; }

  void rewrite(
      ConstructOp op, Binding type, OpAdaptor adaptor, BindingsAdaptor,
      ConversionPatternRewriter &rewriter
  ) const final {
    SmallVector<Value> args =
        llvm::map_to_vector(adaptor.getArgs(), std::bind_front(unwrapToFelt, &rewriter));

    auto replacement = rewriter.replaceOpWithNewOp<Replacement>(
        op, TypeRange({FeltType::get(getContext())}), ValueRange(args), ArrayRef<NamedAttribute>()
    );

    zml::createAndStoreSlot(op, replacement, rewriter, *getTypeConverter());

    // StructDefOp structDef = op->getParentOfType<StructDefOp>();
    // assert(structDef);
    // auto selfOp = op->getParentOfType<zml::SelfOp>();
    // assert(selfOp);
    // auto *slot = type->getSlot();
    // assert(slot);
    // auto *compSlot = mlir::cast<zhl::ComponentSlot>(slot);
    // auto compSlotBinding = compSlot->getBinding();
    //
    // // Create the field
    // FlatSymbolRefAttr name =
    //     zml::createSlot(compSlot, rewriter, structDef, op.getLoc(), *getTypeConverter());
    // Type slotType = zml::materializeTypeBinding(rewriter.getContext(), compSlotBinding);
    //
    // zml::storeSlot(
    //     *compSlot, replacement, name, slotType, op.getLoc(), rewriter, selfOp.getSelfValue()
    // );
  }
};

/// Pattern for ops that can only run in the witness generator that are located in the constrain
/// function. Lowers to operations for loading the result from a field.
class LowerConstructToLlzkWitnessGenFeltOpInConstrain : public LowerConstructToLlzkFeltOpBase {
public:
  using LowerConstructToLlzkFeltOpBase::LowerConstructToLlzkFeltOpBase;

  std::optional<StringRef> scopeFunction() const final { return "constrain"; }

  void rewrite(
      ConstructOp op, Binding type, OpAdaptor, BindingsAdaptor, ConversionPatternRewriter &rewriter
  ) const final {
    StructDefOp structDef = op->getParentOfType<StructDefOp>();
    assert(structDef);
    auto selfOp = op->getParentOfType<zml::SelfOp>();
    assert(selfOp);
    // What about globals??
    auto *slot = type->getSlot();
    assert(slot);
    auto *compSlot = mlir::cast<zhl::ComponentSlot>(slot);
    auto compSlotBinding = compSlot->getBinding();

    FlatSymbolRefAttr name =
        FlatSymbolRefAttr::get(rewriter.getStringAttr(compSlot->getSlotName()));
    Type slotType = zml::materializeTypeBinding(rewriter.getContext(), compSlotBinding);

    auto value = zml::loadSlot(
        *compSlot, llzk::felt::FeltType::get(getContext()), name, slotType, op.getLoc(), rewriter,
        selfOp.getSelfValue()
    );

    rewriter.replaceOp(op, value);
  }
};

} // namespace

void zklang::populateZhlToLlzkFeltConversionPatterns(
    const TypeConverter &tc, RewritePatternSet &patterns
) {
  auto *ctx = patterns.getContext();
  patterns
      // clang-format off
      .add<LowerLiteralToLlzkFeltOp>(tc, ctx)
      .add<LowerConstructToLlzkWitnessGenFeltOpInCompute<AndFeltOp>>("BitAnd", tc, ctx) 
      .add<LowerConstructToLlzkWitnessGenFeltOpInConstrain>("BitAnd", tc, ctx) 
      .add<LowerConstructToLlzkFeltOp<AddFeltOp>>("Add", tc, ctx)
      .add<LowerConstructToLlzkFeltOp<SubFeltOp>>("Sub", tc, ctx)
      .add<LowerConstructToLlzkFeltOp<MulFeltOp>>("Mul", tc, ctx)
      .add<LowerConstructToLlzkFeltOp<ModFeltOp>>("Mod", tc, ctx)
      .add<LowerConstructToLlzkWitnessGenFeltOpInCompute<InvFeltOp>>("Inv", tc, ctx)
      .add<LowerConstructToLlzkWitnessGenFeltOpInConstrain>("Inv", tc, ctx) 
      .add<LowerConstructToLlzkFeltOp<NegFeltOp>>("Neg", tc, ctx)
      // clang-format on
      ;
}

void zklang::populateZhlToLlzkFeltConversionTarget(ConversionTarget &target) {
  target.addLegalDialect<
      ZhlDialect, StructDialect, FeltDialect, FunctionDialect, zml::ZMLDialect, BuiltinDialect>();
  target.addIllegalOp<LiteralOp>();
  target.addDynamicallyLegalOp<ConstructOp>([](ConstructOp op) {
    return !isTargetConstructOp(op, {"BitAnd", "Add", "Sub", "Mul", "Mod", "Inv", "Neg"});
  });
}

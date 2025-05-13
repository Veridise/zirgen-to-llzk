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

#include <zklang/Passes/ConvertZhlToLlzkStruct/Patterns.h>

#include <llvm/ADT/SmallVectorExtras.h>
#include <llzk/Dialect/Function/IR/Dialect.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Dialect.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/LLZK/Builder.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/Dialect.h>

using namespace zklang;
using namespace mlir;
using namespace zirgen::Zhl;
using namespace llzk::component;
using namespace llzk::function;

namespace {

//===----------------------------------------------------------------------===//
// LowerComponentToLlzkStruct
//===----------------------------------------------------------------------===//

class LowerComponentToLlzkStruct : public OpConversionPattern<ComponentOp> {
public:
  using OpConversionPattern<ComponentOp>::OpConversionPattern;

  /// Match only components that have the type binding attribute and that are not externs
  LogicalResult match(ComponentOp op) const override {
    auto typeBinding = op->getAttrOfType<zml::FixedTypeBindingAttr>("zml.binding");
    if (!typeBinding) {
      return failure();
    }
    return failure(typeBinding.getExternComponent());
  }

  Location findSuperLocation(ComponentOp op) const {
    auto &blocks = op.getRegion().getBlocks();
    if (!blocks.empty()) {

      auto term = blocks.front().getTerminator();
      if (term) {
        return term->getLoc();
      }
    }
    return op.getLoc();
  }

  void rewrite(ComponentOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto tc = getTypeConverter();
    assert(tc);
    auto typeBinding = op->getAttrOfType<zml::FixedTypeBindingAttr>("zml.binding");
    llzk::ComponentBuilder builder;
    auto genericNames =
        llvm::map_to_vector(typeBinding.getGenericParamNames(), [](auto s) { return s.str(); });
    auto paramLocations =
        llvm::map_to_vector(typeBinding.getParamLocs(), [](auto l) -> Location { return l; });

    auto super = tc->convertType(typeBinding.getSuperType().getType());
    auto superLoc = findSuperLocation(op);

    builder.name(typeBinding.getName())
        .location(op->getLoc())
        .attrs(op->getAttrs())
        .typeParams(genericNames)
        .constructor(typeBinding.getCtorType(), paramLocations)
        .takeRegion(&op.getRegion())
        .field("$super", super, superLoc);
    auto comp = builder.build(rewriter, *tc);

    rewriter.replaceOp(op.getOperation(), comp.getOperation());
  }
};

//===----------------------------------------------------------------------===//
// LowerExternComponentToLlzkFunction
//===----------------------------------------------------------------------===//

class LowerExternComponentToLlzkFunction : public OpConversionPattern<ComponentOp> {
public:
  using OpConversionPattern<ComponentOp>::OpConversionPattern;

  /// Match only components that have the type binding attribute and that are externs
  LogicalResult
  matchAndRewrite(ComponentOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto typeBinding = op->getAttrOfType<zml::FixedTypeBindingAttr>("zml.binding");
    if (!typeBinding || !typeBinding.getExternComponent()) {
      return failure();
    }
    SmallVector<Type> inputs;
    auto *tc = getTypeConverter();
    auto ctor = typeBinding.getCtorType();
    if (failed(tc->convertTypes(ctor.getInputs(), inputs))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<FuncDefOp>(
        op, op.getName(),
        rewriter.getFunctionType(inputs, {tc->convertType(typeBinding.getSuperType().getType())}),
        ArrayRef({rewriter.getNamedAttr("sym_visibility", rewriter.getStringAttr("private"))}),
        ArrayRef<DictionaryAttr>({})
    );

    return success();
  }
};

} // namespace

void zklang::populateZhlToLlzkStructConversionPatterns(
    const mlir::TypeConverter &tc, mlir::RewritePatternSet &patterns
) {
  patterns.add<
      // clang-format off
      LowerComponentToLlzkStruct, 
      LowerExternComponentToLlzkFunction
      // clang-format on
      >(tc, patterns.getContext());
}

void zklang::populateZhlToLlzkStructConversionTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<ZhlDialect, StructDialect, FunctionDialect, zml::ZMLDialect>();
  target.addIllegalOp<ComponentOp>();
  target.addLegalDialect<BuiltinDialect>();
}

void zklang::populateZhlToLlzkStructConversionPatternsAndLegality(
    const mlir::TypeConverter &tc, mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target
) {
  populateZhlToLlzkStructConversionPatterns(tc, patterns);
  populateZhlToLlzkStructConversionTarget(target);
}

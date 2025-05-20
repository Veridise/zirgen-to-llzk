//===- Pass.cpp - ZHL->ZML conversion pass ----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/LLZK/LLZKTypeConverter.h>
#include <zklang/Dialect/ZHL/Typing/Analysis.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/Typing/ZMLTypeConverter.h>
#include <zklang/Dialect/ZML/Utils/Helpers.h>
#include <zklang/FiniteFields/BabyBear.h>
#include <zklang/Passes/ConvertZhlToZml/Pass.h>
#include <zklang/Passes/ConvertZhlToZml/Patterns.h>

using namespace mlir;

namespace zklang {

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertZhlToZmlPass() {
  return std::make_unique<zml::ConvertZhlToZmlPass>();
}

} // namespace zklang

namespace zml {

namespace {

void createPODComponentsFromClosures(
    zhl::ZIRTypeAnalysis &typeAnalysis, OpBuilder &builder, SymbolTable &st, Block *insertionPoint,
    const mlir::TypeConverter &tc
) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(insertionPoint);
  for (auto *clo : typeAnalysis.getClosures()) {
    assert(clo && clo->hasClosure());
    createPODComponent(*clo, builder, st, tc);
  }
}

} // namespace

void ConvertZhlToZmlPass::runOnOperation() {
  auto &typeAnalysis = getAnalysis<zhl::ZIRTypeAnalysis>();
  if (failed(typeAnalysis)) {
    signalPassFailure();
    return;
  }
  mlir::SmallVector<mlir::Attribute> builtinOverrideSet;
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = module->getContext();

  OpBuilder builder(module);
  SymbolTable st(module);
  mlir::ModuleOp globalsModule = mlir::ModuleOp::create(builder.getUnknownLoc(), "globals");
  st.insert(globalsModule, module.begin());
  // TODO: Add option to configure this field
  ff::babybear::Field BabyBear;
  llzk::LLZKTypeConverter llzkTC(BabyBear);
  createPODComponentsFromClosures(typeAnalysis, builder, st, &module.getRegion().front(), llzkTC);

  ZMLTypeConverter typeConverter;
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<
      ZhlGlobalRemoval, /*ZhlDefineLowering,*/ /*ZhlParameterLowering,*/ ZhlConstructLowering,
      /*ZhlExternLowering,*/ /*ZhlLiteralLowering,*/
      ZhlDeclarationRemoval, /*ZhlSuperLoweringInFunc,*/
      ZhlConstrainLowering, /*ZhlLookupLowering,*/ ZhlArrayLowering, ZhlSubscriptLowering,
      ZhlRangeOpLowering, ZhlMapLowering, ZhlSuperLoweringInMap, ZhlLiteralStrLowering,
      ZhlSuperLoweringInBlock, ZhlBlockLowering, ZhlGenericRemoval, ZhlSpecializeRemoval,
      ZhlReduceLowering, ZhlSwitchLowering, ZhlSuperLoweringInSwitch, ZhlDirectiveRemoval,
      ZhlBackLowering>(typeAnalysis, typeConverter, ctx);
  patterns.add<ZhlConstructGlobalLowering, ZhlGetGlobalLowering>(
      globalsModule, typeAnalysis, typeConverter, ctx
  );

  mlir::ConversionTarget target(*ctx);
  target.addIllegalDialect<zirgen::Zhl::ZhlDialect>();
  target.addLegalDialect<
      ZMLDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::index::IndexDialect,
      mlir::arith::ArithDialect>();

  target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();

  if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
  if (!builtinOverrideSet.empty()) {
    module->setAttr("builtinOverrideSet", mlir::ArrayAttr::get(ctx, builtinOverrideSet));
  }
}

} // namespace zml

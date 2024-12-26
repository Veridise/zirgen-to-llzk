// Copyright 2024 Veridise, Inc.

#include "Pass.h"
#include "Patterns.h"
#include "ZirToZkir/Dialect/ZHL/Typing/Analysis.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/Typing/ZMIRTypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;

namespace zkc {

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertZhlToZmirPass() {
  return std::make_unique<ConvertZhlToZmirPass>();
}

void ConvertZhlToZmirPass::runOnOperation() {
  auto &typeAnalysis = getAnalysis<zhl::ZIRTypeAnalysis>();
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = module->getContext();

  // Init patterns for this transformation
  Zmir::ZMIRTypeConverter typeConverter;
  /*mlir::TypeConverter typeConverter;*/
  /*typeConverter.addConversion([](Type t) { return t; });*/
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<
      ZhlGlobalRemoval, ZhlCompToZmirCompPattern, ZhlDefineLowering, ZhlParameterLowering,
      ZhlConstructLowering, ZhlExternLowering, ZhlLiteralLowering, ZhlDeclarationRemoval,
      ZhlSuperLoweringInFunc, ZhlConstrainLowering, ZhlLookupLowering, ZhlArrayLowering,
      ZhlSubscriptLowering, ZhlRangeOpLowering, ZhlMapLowering, ZhlSuperLoweringInMap,
      ZhlSuperLoweringInBlock, ZhlBlockLowering>(typeAnalysis, ctx);

  // Set conversion target
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<
      zkc::Zmir::ZmirDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect,
      mlir::index::IndexDialect, zirgen::Zhl::ZhlDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();
  target.addIllegalOp<
      zirgen::Zhl::ComponentOp, zirgen::Zhl::ConstructorParamOp, zirgen::Zhl::LiteralOp,
      zirgen::Zhl::ConstructOp, zirgen::Zhl::DefinitionOp, zirgen::Zhl::ConstraintOp,
      zirgen::Zhl::DeclarationOp, zirgen::Zhl::ExternOp, zirgen::Zhl::SuperOp,
      zirgen::Zhl::GlobalOp, zirgen::Zhl::LookupOp, zirgen::Zhl::ArrayOp, zirgen::Zhl::SubscriptOp,
      zirgen::Zhl::RangeOp, zirgen::Zhl::MapOp, zirgen::Zhl::SuperOp, zirgen::Zhl::BlockOp>();

  // Call partialTransformation
  if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
  markAnalysesPreserved<zhl::ZIRTypeAnalysis>();
}

void ConvertZhlToScfPass::runOnOperation() {
  Zmir::ComponentOp op = getOperation();
  // Injected builtins don't need to be converted
  if (op.getBuiltin()) {
    return;
  }

  mlir::MLIRContext *ctx = op->getContext();
  Zmir::ZMIRTypeConverter typeConverter;
  // Init patterns for this transformation
  mlir::RewritePatternSet patterns(ctx);

  /*patterns.add<*/
  /*    >(typeConverter, ctx);*/

  // Set conversion target
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<
      zkc::Zmir::ZmirDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect,
      mlir::index::IndexDialect, zirgen::Zhl::ZhlDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();

  target.addIllegalOp<
      zirgen::Zhl::RangeOp, zirgen::Zhl::MapOp, zirgen::Zhl::SuperOp, zirgen::Zhl::BlockOp>();

  // Call partialTransformation
  if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns)))) {
    signalPassFailure();
  }
  markAnalysesPreserved<zhl::ZIRTypeAnalysis>();
}

std::unique_ptr<OperationPass<Zmir::ComponentOp>> createConvertZhlToScfPass() {
  return std::make_unique<ConvertZhlToScfPass>();
}
} // namespace zkc

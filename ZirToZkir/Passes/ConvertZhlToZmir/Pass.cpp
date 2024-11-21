// Copyright 2024 Veridise, Inc.

#include "Pass.h"
#include "Patterns.h"
#include "ZMIRTypeConverter.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Passes/ConvertZhlToZmir/ZMIRTypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;

namespace zkc {

void ConvertZhlToZmirPass::runOnOperation() {
  Zmir::ComponentOp op = getOperation();
  // Injected builtins don't need to be converted
  if (op.getBuiltin()) {
    return;
  }

  mlir::MLIRContext *ctx = op->getContext();
  Zmir::ZMIRTypeConverter typeConverter;
  // Init patterns for this transformation
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<ZhlLiteralLowering, ZhlLiteralStrLowering, ZhlParameterLowering,
               ZhlConstructLowering, ZhlDefineLowering, ZhlConstrainLowering,
               ZhlSuperLowering, ZhlGlobalRemoval, ZhlDeclarationRemoval,
               ZhlExternLowering, ZhlSubscriptLowering, ZhlLookupLowering,
               ZhlArrayLowering>(typeConverter, ctx);
  // Set conversion target
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<zkc::Zmir::ZmirDialect, mlir::func::FuncDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();
  target.addIllegalDialect<zirgen::Zhl::ZhlDialect>();

  // Call partialTransformation
  if (mlir::failed(
          mlir::applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<Zmir::ComponentOp>> createConvertZhlToZmirPass() {
  return std::make_unique<ConvertZhlToZmirPass>();
}

void TransformComponentDeclsPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = module->getContext();

  // Init patterns for this transformation
  Zmir::ZMIRTypeConverter typeConverter;
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<ZhlCompToZmirCompPattern>(typeConverter, ctx);

  // Set conversion target
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<zkc::Zmir::ZmirDialect, mlir::func::FuncDialect>();
  target.addIllegalOp<zirgen::Zhl::ComponentOp>();

  // Call partialTransformation
  if (mlir::failed(
          mlir::applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createTransformComponentDeclsPass() {
  return std::make_unique<TransformComponentDeclsPass>();
}
} // namespace zkc

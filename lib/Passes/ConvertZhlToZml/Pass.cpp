// Copyright 2024 Veridise, Inc.

#include "zklang/Passes/ConvertZhlToZml/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Dialect/ZHL/Typing/Analysis.h"
#include "zklang/Dialect/ZML/IR/Dialect.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include "zklang/Dialect/ZML/Typing/ZMIRTypeConverter.h"
#include "zklang/Passes/ConvertZhlToZml/Patterns.h"
#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;

namespace zkc {

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertZhlToZmirPass() {
  return std::make_unique<ConvertZhlToZmirPass>();
}

void ConvertZhlToZmirPass::runOnOperation() {
  auto &typeAnalysis = getAnalysis<zhl::ZIRTypeAnalysis>();
  mlir::SmallVector<mlir::Attribute> builtinOverrideSet;
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = module->getContext();

  // Init patterns for this transformation
  Zmir::ZMIRTypeConverter typeConverter;
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<
      ZhlGlobalRemoval, ZhlDefineLowering, ZhlParameterLowering, ZhlConstructLowering,
      ZhlExternLowering, ZhlLiteralLowering, ZhlDeclarationRemoval, ZhlSuperLoweringInFunc,
      ZhlConstrainLowering, ZhlLookupLowering, ZhlArrayLowering, ZhlSubscriptLowering,
      ZhlRangeOpLowering, ZhlMapLowering, ZhlSuperLoweringInMap, ZhlLiteralStrLowering,
      ZhlSuperLoweringInBlock, ZhlBlockLowering, ZhlGenericRemoval, ZhlSpecializeRemoval,
      ZhlReduceLowering, ZhlSwitchLowering, ZhlSuperLoweringInSwitch, ZhlDirectiveRemoval>(
      typeAnalysis, typeConverter, ctx
  );
  patterns.add<ZhlCompToZmirCompPattern>([&](mlir::StringRef name) {
    builtinOverrideSet.push_back(mlir::StringAttr::get(ctx, name));
  }, typeAnalysis, typeConverter, ctx);

  // Set conversion target
  mlir::ConversionTarget target(*ctx);
  target.addIllegalDialect<zirgen::Zhl::ZhlDialect>();
  target.addLegalDialect<
      zkc::Zmir::ZmirDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect,
      mlir::index::IndexDialect, mlir::arith::ArithDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();

  // Call partialTransformation
  if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
  if (!builtinOverrideSet.empty()) {
    module->setAttr("builtinOverrideSet", mlir::ArrayAttr::get(ctx, builtinOverrideSet));
  }
}

} // namespace zkc

// Copyright 2024 Veridise, Inc.

#include "zklang/Passes/ConvertZhlToZml/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Dialect/ZHL/Typing/Analysis.h"
#include "zklang/Dialect/ZML/IR/Dialect.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include "zklang/Dialect/ZML/Typing/ZMLTypeConverter.h"
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

namespace zklang {

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertZhlToZmlPass() {
  return std::make_unique<zml::ConvertZhlToZmlPass>();
}

} // namespace zklang

namespace zml {

void ConvertZhlToZmlPass::runOnOperation() {
  auto &typeAnalysis = getAnalysis<zhl::ZIRTypeAnalysis>();
  mlir::SmallVector<mlir::Attribute> builtinOverrideSet;
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = module->getContext();

  // Init patterns for this transformation
  ZMLTypeConverter typeConverter;
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
  target.addLegalDialect<
      ZMLDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::index::IndexDialect,
      zirgen::Zhl::ZhlDialect, mlir::arith::ArithDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();
  target.addIllegalOp<
      zirgen::Zhl::ComponentOp, zirgen::Zhl::ConstructorParamOp, zirgen::Zhl::LiteralOp,
      zirgen::Zhl::ConstructOp, zirgen::Zhl::DefinitionOp, zirgen::Zhl::ConstraintOp,
      zirgen::Zhl::DeclarationOp, zirgen::Zhl::ExternOp, zirgen::Zhl::SuperOp,
      zirgen::Zhl::GlobalOp, zirgen::Zhl::LookupOp, zirgen::Zhl::ArrayOp, zirgen::Zhl::SubscriptOp,
      zirgen::Zhl::RangeOp, zirgen::Zhl::MapOp, zirgen::Zhl::SuperOp, zirgen::Zhl::BlockOp,
      zirgen::Zhl::StringOp, zirgen::Zhl::TypeParamOp, zirgen::Zhl::SpecializeOp,
      zirgen::Zhl::ReduceOp, zirgen::Zhl::SwitchOp, zirgen::Zhl::DirectiveOp>();

  // Call partialTransformation
  if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
  if (!builtinOverrideSet.empty()) {
    module->setAttr("builtinOverrideSet", mlir::ArrayAttr::get(ctx, builtinOverrideSet));
  }
}

} // namespace zml

#include <algorithm>
#include <cassert>
#include <iterator>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llzk/Dialect/LLZK/IR/Dialect.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <llzk/Dialect/LLZK/Util/SymbolHelper.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <unordered_set>
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/Utils/Patterns.h>
#include <zklang/Passes/ConvertZmlToLlzk/LLZKTypeConverter.h>
#include <zklang/Passes/ConvertZmlToLlzk/Pass.h>
#include <zklang/Passes/ConvertZmlToLlzk/Patterns.h>

using namespace mlir;

namespace zml {

void ConvertZmlToLlzkPass::runOnOperation() {
  auto op = getOperation();

  mlir::MLIRContext *ctx = op->getContext();
  llzk::LLZKTypeConverter typeConverter;

  mlir::RewritePatternSet patterns(ctx);

  patterns.add<
      LitValOpLowering, ReplaceSelfWith<NewOp<llzk::CreateStructOp>>, LowerBitAnd, LowerAdd,
      LowerSub, LowerMul, LowerInv, LowerIsz, LowerNeg, LowerConstrainOp, LowerReadFieldOp,
      LowerInRangeOp, LowerNewArrayOp, LowerReadArrayOp, LowerAllocArrayOp, LowerArrayLengthOp,
      LowerIndexToValOp, LowerValToIndexOp, LowerWriteArrayOp, WriteFieldOpLowering,
      LowerConstrainCallOp, LowerNopOp, LowerSuperCoerceOp, LowerMod, LowerLoadValParamOp,
      ComponentLowering, FieldDefOpLowering, FuncOpLowering, ReturnOpLowering, ExternCallOpLowering,
      CallIndirectOpLoweringInCompute, WriteFieldOpLowering, RemoveConstructorRefOp,
      RemoveExternFnRefOp,           /*UpdateScfForOpTypes, UpdateScfYieldOpTypes,*/
      UpdateScfExecuteRegionOpTypes, /* UpdateScfIfOpTypes,*/
      ValToI1OpLowering, AssertOpLowering, LowerLitValArrayOp

      >(typeConverter, ctx);

  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<llzk::LLZKDialect, mlir::arith::ArithDialect, index::IndexDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();
  target.addIllegalDialect<ZMLDialect, mlir::func::FuncDialect>();

  // Legality for scf ops added manually here because ExecuteRegionOp and its associated YieldOp are
  // not supported by the MLIR provided populate* function
  target.addDynamicallyLegalOp<scf::ExecuteRegionOp>([&](scf::ExecuteRegionOp execOp) {
    return typeConverter.isLegal(execOp.getResultTypes());
  });

  target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp yieldOp) {
    if (!isa<scf::ExecuteRegionOp, scf::ForOp, scf::IfOp, scf::WhileOp>(yieldOp->getParentOp())) {
      return true;
    }
    return typeConverter.isLegal(yieldOp.getOperandTypes());
  });

  target.addDynamicallyLegalOp<scf::ForOp, scf::IfOp>([&](Operation *scfOp) {
    return typeConverter.isLegal(scfOp->getResultTypes());
  });

  target.addDynamicallyLegalOp<scf::WhileOp, scf::ConditionOp>([&](Operation *scfOp) {
    return typeConverter.isLegal(scfOp);
  });

  scf::populateSCFStructuralTypeConversions(typeConverter, patterns);

  llvm::dbgs() << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaa\n";
  if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns)))) {
    llvm::dbgs(
    ) << "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n";
    signalPassFailure();
  }
  llvm::dbgs() << "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n";
}

void InjectLlzkModAttrsPass::runOnOperation() {
  auto op = getOperation();
  op->setAttr(
      llzk::LANG_ATTR_NAME,
      mlir::StringAttr::get(&getContext(), llzk::LLZKDialect::getDialectNamespace())
  );
}

} // namespace zml

namespace zklang {

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertZmlToLlzkPass() {
  return std::make_unique<zml::ConvertZmlToLlzkPass>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> createInjectLlzkModAttrsPass() {
  return std::make_unique<zml::InjectLlzkModAttrsPass>();
}

} // namespace zklang

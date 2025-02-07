#include "zklang/Passes/ConvertZmlToLlzk/Pass.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zklang/Dialect/ZML/IR/Dialect.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include "zklang/Passes/ConvertZmlToLlzk/LLZKTypeConverter.h"
#include "zklang/Passes/ConvertZmlToLlzk/Patterns.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <unordered_set>
#include <zklang/Dialect/ZML/Utils/Patterns.h>

using namespace mlir;
using namespace zkc::Zmir;

namespace zkc {

void ConvertZmlToLlzkPass::runOnOperation() {
  auto op = getOperation();

  mlir::MLIRContext *ctx = op->getContext();
  llzk::LLZKTypeConverter typeConverter;

  // Init patterns for this transformation
  mlir::RewritePatternSet patterns(ctx);

  patterns.add<
      LitValOpLowering, ReplaceSelfWith<NewOp<llzk::CreateStructOp>>, LowerBitAnd, LowerAdd,
      LowerSub, LowerMul, LowerInv, LowerIsz, LowerNeg, LowerConstrainOp, LowerReadFieldOp,
      LowerInRangeOp, LowerNewArrayOp, LowerReadArrayOp, LowerAllocArrayOp, LowerArrayLengthOp,
      LowerIndexToValOp, LowerValToIndexOp, LowerWriteArrayOp, WriteFieldOpLowering,
      LowerConstrainCallOp, LowerNopOp, LowerSuperCoerceOp, LowerMod, LowerLoadValParamOp,
      ComponentLowering, FieldDefOpLowering, FuncOpLowering, ReturnOpLowering, CallOpLowering,
      CallIndirectOpLoweringInCompute, WriteFieldOpLowering, RemoveConstructorRefOp,
      UpdateScfForOpTypes, UpdateScfYieldOpTypes, UpdateScfExecuteRegionOpTypes>(
      typeConverter, ctx
  );

  // Set conversion target
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<llzk::LLZKDialect, mlir::arith::ArithDialect, index::IndexDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();
  target.addIllegalDialect<zkc::Zmir::ZmirDialect, mlir::func::FuncDialect>();

  // Control flow operations may need to update their types
  target.addDynamicallyLegalDialect<scf::SCFDialect>([&](Operation *scfOp) {
    return typeConverter.isLegal(scfOp);
  });

  // Call partialTransformation
  if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertZmlToLlzkPass() {
  return std::make_unique<ConvertZmlToLlzkPass>();
}

void InjectLlzkModAttrsPass::runOnOperation() {
  auto op = getOperation();
  op->setAttr(
      llzk::LANG_ATTR_NAME,
      mlir::StringAttr::get(&getContext(), llzk::LLZKDialect::getDialectNamespace())
  );
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> createInjectLlzkModAttrsPass() {
  return std::make_unique<InjectLlzkModAttrsPass>();
}

} // namespace zkc

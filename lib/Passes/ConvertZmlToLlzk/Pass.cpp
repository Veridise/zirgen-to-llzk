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
#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace zkc::Zmir;

namespace zkc {

void ConvertZmirToLlzkPass::runOnOperation() {
  auto op = getOperation();

  mlir::MLIRContext *ctx = op->getContext();
  llzk::LLZKTypeConverter typeConverter;
  // Init patterns for this transformation
  mlir::RewritePatternSet patterns(ctx);

  patterns.add<
      LitValOpLowering, GetSelfOpLowering, LowerBitAnd, LowerAdd, LowerSub, LowerMul, LowerInv,
      LowerIsz, LowerNeg, LowerConstrainOp, LowerReadFieldOp, LowerInRangeOp, LowerNewArrayOp,
      LowerReadArrayOp, LowerAllocArrayOp, LowerArrayLengthOp, LowerIndexToValOp, LowerValToIndexOp,
      LowerWriteArrayOp, WriteFieldOpLowering, LowerConstrainCallOp, LowerNopOp, LowerSuperCoerceOp,
      LowerLoadValParamOp>(typeConverter, ctx);

  // Set conversion target
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<
      llzk::LLZKDialect, mlir::arith::ArithDialect, index::IndexDialect, scf::SCFDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();
  target.addIllegalDialect<zkc::Zmir::ZmirDialect, mlir::func::FuncDialect>();

  target.addIllegalOp<
      LitValOp, GetSelfOp, BitAndOp, AddOp, SubOp, MulOp, InvOp, IsZeroOp, NegOp, ReadFieldOp,
      ConstrainOp, InRangeOp, NewArrayOp, ReadArrayOp, AllocArrayOp, GetArrayLenOp, IndexToValOp,
      ValToIndexOp, WriteArrayOp, WriteFieldOp, ConstrainCallOp, NopOp, SuperCoerceOp,
      LoadValParamOp>();

  // Call partialTransformation
  if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<llzk::StructDefOp>> createConvertZmirToLlzkPass() {
  return std::make_unique<ConvertZmirToLlzkPass>();
}

void ConvertZmirComponentsToLlzkPass::runOnOperation() {
  auto op = getOperation();
  op->setAttr(
      llzk::LANG_ATTR_NAME,
      mlir::StringAttr::get(&getContext(), llzk::LLZKDialect::getDialectNamespace())
  );

  mlir::MLIRContext *ctx = op->getContext();
  llzk::LLZKTypeConverter typeConverter;
  // Init patterns for this transformation
  mlir::RewritePatternSet patterns(ctx);

  patterns.add<
      ComponentLowering, FieldDefOpLowering, FuncOpLowering, ReturnOpLowering, CallOpLowering,
      CallIndirectOpLoweringInCompute, CallIndirectOpLoweringInConstrain, WriteFieldOpLowering,
      RemoveConstructorRefOp>(typeConverter, ctx);

  // Set conversion target
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<
      zkc::Zmir::ZmirDialect, mlir::func::FuncDialect, llzk::LLZKDialect, mlir::arith::ArithDialect,
      index::IndexDialect, scf::SCFDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();

  target.addIllegalOp<
      ComponentOp, SplitComponentOp, FieldDefOp, func::FuncOp, func::ReturnOp, func::CallOp,
      func::CallIndirectOp, WriteFieldOp, ConstructorRefOp>();
  // Call partialTransformation
  if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertZmirComponentsToLlzkPass() {
  return std::make_unique<ConvertZmirComponentsToLlzkPass>();
}

} // namespace zkc

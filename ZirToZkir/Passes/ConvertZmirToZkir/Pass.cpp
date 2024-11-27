#include "Pass.h"
#include "Patterns.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Passes/ConvertZmirToZkir/ZKIRTypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zkir/Dialect/ZKIR/IR/Dialect.h"
#include "zkir/Dialect/ZKIR/IR/Ops.h"
#include "zkir/Dialect/ZKIR/Util/SymbolHelper.h"
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

void ConvertZmirToZkirPass::runOnOperation() {
  auto op = getOperation();

  mlir::MLIRContext *ctx = op->getContext();
  zkir::ZKIRTypeConverter typeConverter;
  // Init patterns for this transformation
  mlir::RewritePatternSet patterns(ctx);

  patterns.add<LitValOpLowering, GetSelfOpLowering>(typeConverter, ctx);

  // Set conversion target
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<zkc::Zmir::ZmirDialect, mlir::func::FuncDialect,
                         zkir::ZKIRDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();

  target.addIllegalOp<LitValOp, GetSelfOp>();

  // Call partialTransformation
  if (mlir::failed(
          mlir::applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<zkir::StructDefOp>>
createConvertZmirToZkirPass() {
  return std::make_unique<ConvertZmirToZkirPass>();
}

void ConvertZmirComponentsToZkirPass::runOnOperation() {
  auto op = getOperation();
  op->setAttr(zkir::LANG_ATTR_NAME,
              mlir::StringAttr::get(&getContext(),
                                    zkir::ZKIRDialect::getDialectNamespace()));

  mlir::MLIRContext *ctx = op->getContext();
  zkir::ZKIRTypeConverter typeConverter;
  // Init patterns for this transformation
  mlir::RewritePatternSet patterns(ctx);

  patterns
      .add<ComponentLowering, FieldDefOpLowering, FuncOpLowering,
           ReturnOpLowering, CallOpLowering, CallIndirectOpLoweringInCompute,
           CallIndirectOpLoweringInConstrain, WriteFieldOpLowering,
           RemoveConstructorRefOp, LowerBitAnd, LowerAdd, LowerSub, LowerMul,
           LowerInv, LowerNeg, LowerConstrainOp, LowerReadFieldOp,
           LowerInRangeOp>(typeConverter, ctx);

  // Set conversion target
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<zkc::Zmir::ZmirDialect, mlir::func::FuncDialect,
                         zkir::ZKIRDialect, mlir::arith::ArithDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();

  target
      .addIllegalOp<ComponentOp, SplitComponentOp, FieldDefOp, func::FuncOp,
                    func::ReturnOp, func::CallOp, func::CallIndirectOp,
                    WriteFieldOp, ConstructorRefOp, BitAndOp, AddOp, SubOp,
                    MulOp, InvOp, NegOp, ReadFieldOp, ConstrainOp, InRangeOp>();
  // Call partialTransformation
  if (mlir::failed(
          mlir::applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConvertZmirComponentsToZkirPass() {
  return std::make_unique<ConvertZmirComponentsToZkirPass>();
}

} // namespace zkc

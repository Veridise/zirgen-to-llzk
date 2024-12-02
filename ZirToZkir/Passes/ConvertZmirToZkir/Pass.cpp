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

  patterns
      .add<LitValOpLowering, GetSelfOpLowering, LowerBitAnd, LowerAdd, LowerSub,
           LowerMul, LowerInv, LowerIsz, LowerNeg, LowerConstrainOp,
           LowerReadFieldOp, LowerInRangeOp, LowerNewArrayOp, LowerReadArrayOp,
           LowerAllocArrayOp, LowerArrayLengthOp, LowerIndexToValOp,
           LowerValToIndexOp, LowerWriteArrayOp, WriteFieldOpLowering>(
          typeConverter, ctx);

  // Set conversion target
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<zkir::ZKIRDialect, mlir::arith::ArithDialect,
                         index::IndexDialect, scf::SCFDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();
  target.addIllegalDialect<zkc::Zmir::ZmirDialect, mlir::func::FuncDialect>();

  target.addIllegalOp<LitValOp, GetSelfOp, BitAndOp, AddOp, SubOp, MulOp, InvOp,
                      IsZeroOp, NegOp, ReadFieldOp, ConstrainOp, InRangeOp,
                      NewArrayOp, ReadArrayOp, AllocArrayOp, GetArrayLenOp,
                      IndexToValOp, ValToIndexOp, WriteArrayOp, WriteFieldOp>();

  // Call partialTransformation
  if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns))))
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
           RemoveConstructorRefOp>(typeConverter, ctx);

  // Set conversion target
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<zkc::Zmir::ZmirDialect, mlir::func::FuncDialect,
                         zkir::ZKIRDialect, mlir::arith::ArithDialect,
                         index::IndexDialect, scf::SCFDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();

  target.addIllegalOp<ComponentOp, SplitComponentOp, FieldDefOp, func::FuncOp,
                      func::ReturnOp, func::CallOp, func::CallIndirectOp,
                      WriteFieldOp, ConstructorRefOp>();
  // Call partialTransformation
  if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConvertZmirComponentsToZkirPass() {
  return std::make_unique<ConvertZmirComponentsToZkirPass>();
}

} // namespace zkc

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

using namespace mlir;
using namespace zkc::Zmir;

namespace zkc {

void ConvertZmlToLlzkPass::runOnOperation() {
  auto op = getOperation();

  mlir::MLIRContext *ctx = op->getContext();
  // mlir::ModuleOp mod = op->getParentOfType<mlir::ModuleOp>();
  // if (mod->hasAttrOfType<mlir::ArrayAttr>("builtinOverrideSet")) {
  //   auto attrSet = mod->getAttrOfType<mlir::ArrayAttr>("builtinOverrideSet");
  //   std::transform(
  //       attrSet.begin(), attrSet.end(), std::inserter(builtinOverrideSet,
  //       builtinOverrideSet.end()),
  //       [](auto attr) {
  //     auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
  //     assert(strAttr && "attribute elements in builtinOverrideSet must be strings");
  //     return strAttr.getValue();
  //   }
  //   );
  // }
  llzk::LLZKTypeConverter typeConverter(op);

  // Init patterns for this transformation
  mlir::RewritePatternSet patterns(ctx);

  patterns.add<
      LitValOpLowering, GetSelfOpLowering, LowerBitAnd, LowerAdd, LowerSub, LowerMul, LowerInv,
      LowerIsz, LowerNeg, LowerConstrainOp, LowerReadFieldOp, LowerInRangeOp, LowerNewArrayOp,
      LowerReadArrayOp, LowerAllocArrayOp, LowerArrayLengthOp, LowerIndexToValOp, LowerValToIndexOp,
      LowerWriteArrayOp, WriteFieldOpLowering, LowerConstrainCallOp, LowerNopOp, LowerSuperCoerceOp,
      LowerMod, LowerLoadValParamOp, ComponentLowering, FieldDefOpLowering, FuncOpLowering,
      ReturnOpLowering, CallOpLowering, CallIndirectOpLoweringInCompute,
      CallIndirectOpLoweringInConstrain, WriteFieldOpLowering, RemoveConstructorRefOp,
      UpdateScfForOpTypes, UpdateScfYieldOpTypes, UpdateScfExecuteRegionOpTypes>(
      typeConverter, ctx
  );

  // Set conversion target
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<
      llzk::LLZKDialect, mlir::arith::ArithDialect, index::IndexDialect /*, scf::SCFDialect*/>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();
  target.addIllegalDialect<zkc::Zmir::ZmirDialect, mlir::func::FuncDialect>();

  // Control flow operations may need to update their types
  target.addDynamicallyLegalDialect<scf::SCFDialect>([&](Operation *scfOp) {
    return typeConverter.isLegal(scfOp);
  });

  // target.addIllegalOp<
  //     LitValOp, GetSelfOp, BitAndOp, AddOp, SubOp, MulOp, ModOp, InvOp, IsZeroOp, NegOp,
  //     ReadFieldOp, ConstrainOp, InRangeOp, NewArrayOp, ReadArrayOp, AllocArrayOp, GetArrayLenOp,
  //     IndexToValOp, ValToIndexOp, WriteArrayOp, WriteFieldOp, ConstrainCallOp, NopOp,
  //     SuperCoerceOp, LoadValParamOp>();

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

  // std::unordered_set<std::string_view> builtinOverrideSet;
  // if (mlir::isa<mlir::ModuleOp>(op) && op->hasAttrOfType<mlir::ArrayAttr>("builtinOverrideSet"))
  // {
  //   auto attrSet = op->getAttrOfType<mlir::ArrayAttr>("builtinOverrideSet");
  //   std::transform(
  //       attrSet.begin(), attrSet.end(), std::inserter(builtinOverrideSet,
  //       builtinOverrideSet.end()),
  //       [](auto attr) {
  //     auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
  //     assert(strAttr && "attribute elements in builtinOverrideSet must be strings");
  //     return strAttr.getValue();
  //   }
  //   );
  // }
  // mlir::MLIRContext *ctx = op->getContext();
  // llzk::LLZKTypeConverter typeConverter(builtinOverrideSet);
  // // Init patterns for this transformation
  // mlir::RewritePatternSet patterns(ctx);
  //
  // patterns.add<
  //     ComponentLowering, FieldDefOpLowering, FuncOpLowering, ReturnOpLowering, CallOpLowering,
  //     CallIndirectOpLoweringInCompute, CallIndirectOpLoweringInConstrain, WriteFieldOpLowering,
  //     RemoveConstructorRefOp>(typeConverter, ctx);
  //
  // // Set conversion target
  // mlir::ConversionTarget target(*ctx);
  // target.addLegalDialect<
  //     zkc::Zmir::ZmirDialect, mlir::func::FuncDialect, llzk::LLZKDialect,
  //     mlir::arith::ArithDialect, index::IndexDialect, scf::SCFDialect>();
  // target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();
  //
  // target.addIllegalOp<
  //     ComponentOp, SplitComponentOp, FieldDefOp, func::FuncOp, func::ReturnOp, func::CallOp,
  //     func::CallIndirectOp, WriteFieldOp, ConstructorRefOp>();
  // // Call partialTransformation
  // if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns)))) {
  //   signalPassFailure();
  // }
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> createInjectLlzkModAttrsPass() {
  return std::make_unique<InjectLlzkModAttrsPass>();
}

} // namespace zkc

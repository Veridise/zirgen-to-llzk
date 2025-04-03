#include <algorithm>
#include <cassert>
#include <iterator>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llzk/Dialect/LLZK/IR/Dialect.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <llzk/Dialect/LLZK/Util/SymbolHelper.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <unordered_set>
#include <zklang/Dialect/ZML/ExtVal/BabyBear.h>
#include <zklang/Dialect/ZML/ExtVal/Conversion.h>
#include <zklang/Dialect/ZML/ExtVal/Patterns.h>
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/Utils/Patterns.h>
#include <zklang/FiniteFields/BabyBear.h>
#include <zklang/FiniteFields/Field.h>
#include <zklang/Passes/ConvertZmlToLlzk/LLZKTypeConverter.h>
#include <zklang/Passes/ConvertZmlToLlzk/Pass.h>
#include <zklang/Passes/ConvertZmlToLlzk/Patterns.h>

using namespace mlir;

namespace {

template <typename T> std::unique_ptr<ff::FieldData> mkField() { return std::make_unique<T>(); }

using LLZKTypeConverterPtr = std::unique_ptr<llzk::LLZKTypeConverter>;
using ConverterPtr = std::unique_ptr<zml::extval::BaseConverter>;

LogicalResult configureField(
    StringRef name, LLZKTypeConverterPtr &typeConverter, ConverterPtr &converter,
    llvm::function_ref<InFlightDiagnostic()> emitError
) {
  return llvm::StringSwitch<llvm::function_ref<LogicalResult()>>(name)
      .Case("babybear", [&] {
    ff::babybear::Field BabyBear;
    typeConverter = std::make_unique<llzk::LLZKTypeConverter>(BabyBear);
    converter = std::make_unique<zml::extval::babybear::Converter>(*typeConverter);
    return success();
  }).Default([&] { return emitError() << "Unrecognized field " << name; })();
}
} // namespace

namespace zml {

void ConvertZmlToLlzkPass::runOnOperation() {
  ModuleOp op = getOperation();

  mlir::MLIRContext *ctx = &getContext();
  LLZKTypeConverterPtr typeConverter;
  ConverterPtr extValConverter;
  // Only BabyBear is supported for now. More to come (LLZK-180)
  if (failed(configureField(selectedExtValField, typeConverter, extValConverter, [&op] {
    return op->emitError();
  }))) {
    llvm::errs() << "Failed to configure field\n";
    signalPassFailure();
    return;
  }

  mlir::RewritePatternSet patterns(ctx);

  patterns.add<
      LitValOpLowering, ReplaceSelfWith<NewOp<llzk::CreateStructOp>>, LowerBitAnd, LowerAdd,
      LowerSub, LowerMul, LowerInv, LowerIsz, LowerNeg, LowerConstrainOp, LowerReadFieldOp,
      LowerInRangeOp, LowerNewArrayOp, LowerReadArrayOp, LowerAllocArrayOp, LowerArrayLengthOp,
      LowerIndexToValOp, LowerValToIndexOp, LowerWriteArrayOp, WriteFieldOpLowering,
      LowerConstrainCallOp, LowerNopOp, LowerSuperCoerceOp, LowerMod, LowerLoadValParamOp,
      ComponentLowering, FieldDefOpLowering, FuncOpLowering, ReturnOpLowering, ExternCallOpLowering,
      CallIndirectOpLoweringInCompute, RemoveConstructorRefOp, RemoveExternFnRefOp,
      UpdateScfExecuteRegionOpTypes, ValToI1OpLowering, AssertOpLowering, LowerLitValArrayOp,
      LitStrOpLowering, LowerVarArgsOp, LowerGlobalDefOp, LowerSetGlobalOp, LowerGetGlobalOp>(
      *typeConverter, ctx
  );

  populateExtValToLlzkConversionPatterns(patterns, *typeConverter, ctx, *extValConverter);

  scf::populateSCFStructuralTypeConversions(*typeConverter, patterns);

  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<llzk::LLZKDialect, mlir::arith::ArithDialect, index::IndexDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();
  target.addIllegalDialect<ZMLDialect, mlir::func::FuncDialect>();

  // Legality for scf ops added manually here because ExecuteRegionOp and its associated YieldOp are
  // not supported by the MLIR provided populate* function
  target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp yieldOp) {
    if (!isa<scf::ExecuteRegionOp, scf::ForOp, scf::IfOp, scf::WhileOp>(yieldOp->getParentOp())) {
      return true;
    }
    return typeConverter->isLegal(yieldOp.getOperandTypes());
  });

  target.addDynamicallyLegalOp<scf::ExecuteRegionOp, scf::ForOp, scf::IfOp>(
      [&typeConverter](Operation *scfOp) { return typeConverter->isLegal(scfOp->getResultTypes()); }
  );

  target.addDynamicallyLegalOp<scf::WhileOp, scf::ConditionOp>([&typeConverter](Operation *scfOp) {
    return typeConverter->isLegal(scfOp);
  });

  if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

void InjectLlzkModAttrsPass::runOnOperation() {
  ModuleOp op = getOperation();
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

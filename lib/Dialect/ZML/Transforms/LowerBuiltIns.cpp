// Copyright 2024 Veridise, Inc.

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Dialect/ZML/BuiltIns/BuiltIns.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include "zklang/Dialect/ZML/IR/Types.h"
#include "zklang/Dialect/ZML/Transforms/PassDetail.h"
#include "zklang/Dialect/ZML/Typing/ZMLTypeConverter.h"
#include <cassert>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Transforms/DialectConversion.h>
#include <tuple>
#include <unordered_set>

using namespace mlir;

namespace zml {

mlir::FailureOr<ConstructorRefOp> getConstructorRef(func::CallIndirectOp op) {
  auto constrRef = mlir::dyn_cast<ConstructorRefOp>(op.getCallee().getDefiningOp());
  if (!constrRef) {
    return mlir::failure();
  }
  return constrRef;
}

mlir::FailureOr<std::string> getCallIndirectName(func::CallIndirectOp op) {
  auto constrRef = getConstructorRef(op);
  if (mlir::failed(constrRef)) {
    return mlir::failure();
  }
  return constrRef->getComponent().str();
}

bool isLegalCallIndirect(func::CallIndirectOp op) {
  auto name = getCallIndirectName(op);
  if (mlir::failed(name)) {
    return true;
  }
  auto ref = getConstructorRef(op);
  if (mlir::failed(ref)) {
    return true;
  }
  auto found = BuiltInComponentNames.find(*name) != BuiltInComponentNames.end();
  auto markedBuiltin = ref->getBuiltin();
  // Is the constructor call from a component that is built-in?
  return !(found && markedBuiltin);
}

bool isLegalConstructRefOp(ConstructorRefOp op) {
  auto found = BuiltInComponentNames.find(op.getComponent().str()) != BuiltInComponentNames.end();
  auto markedBuiltin = op.getBuiltin();

  return !(found && markedBuiltin);
}

template <typename NewOp, const char *Name>
class ReplaceConstructorCallWithBuiltIn : public OpConversionPattern<func::CallIndirectOp> {
public:
  using OpConversionPattern<func::CallIndirectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::CallIndirectOp op,
      typename OpConversionPattern<func::CallIndirectOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter
  ) const override {
    auto name = getCallIndirectName(op);
    if (mlir::failed(name)) {
      return name;
    }

    // If it's not the one we are looking for just reject the match
    if (*name != Name) {
      return mlir::failure();
    }

    mlir::ValueRange args(
        mlir::iterator_range(adaptor.getOperands().begin() + 1, adaptor.getOperands().end())
    );
    rewriter.replaceOpWithNewOp<NewOp>(op, op.getResult(0).getType(), args);
    return mlir::success();
  }
};

class RemoveConstructorRef : public OpConversionPattern<ConstructorRefOp> {
public:
  using OpConversionPattern<ConstructorRefOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstructorRefOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (isLegalConstructRefOp(op)) {
      return mlir::failure();
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

using BitAndPattern = ReplaceConstructorCallWithBuiltIn<BitAndOp, BitAndStr>;
using AddPattern = ReplaceConstructorCallWithBuiltIn<AddOp, AddStr>;
using SubPattern = ReplaceConstructorCallWithBuiltIn<SubOp, SubStr>;
using MulPattern = ReplaceConstructorCallWithBuiltIn<MulOp, MulStr>;
using ModPattern = ReplaceConstructorCallWithBuiltIn<ModOp, ModStr>;
using InvPattern = ReplaceConstructorCallWithBuiltIn<InvOp, InvStr>;
using IszPattern = ReplaceConstructorCallWithBuiltIn<IsZeroOp, IszStr>;
using NegPattern = ReplaceConstructorCallWithBuiltIn<NegOp, NegStr>;
using InRangePattern = ReplaceConstructorCallWithBuiltIn<InRangeOp, InRangeStr>;

namespace {
class LowerBuiltInsPass : public LowerBuiltInsBase<LowerBuiltInsPass> {

  void runOnOperation() override {
    auto op = getOperation();

    ZMLTypeConverter typeConverter;
    mlir::MLIRContext *ctx = op->getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<
        BitAndPattern, AddPattern, SubPattern, MulPattern, InvPattern, IszPattern, NegPattern,
        ModPattern, InRangePattern, RemoveConstructorRef>(typeConverter, ctx);

    // Set conversion target
    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<
        ZMLDialect, mlir::func::FuncDialect, mlir::index::IndexDialect, mlir::scf::SCFDialect,
        mlir::arith::ArithDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();

    target.addDynamicallyLegalOp<func::CallIndirectOp>(isLegalCallIndirect);
    target.addDynamicallyLegalOp<ConstructorRefOp>(isLegalConstructRefOp);

    // Call partialTransformation
    if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLowerBuiltInsPass() {
  return std::make_unique<LowerBuiltInsPass>();
}

} // namespace zml

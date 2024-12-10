// Copyright 2024 Veridise, Inc.

#include "ZirToZkir/Dialect/ZMIR/BuiltIns/BuiltIns.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include "ZirToZkir/Dialect/ZMIR/Transforms/PassDetail.h"
#include "ZirToZkir/Dialect/ZMIR/Typing/ZMIRTypeConverter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include <cassert>
#include <llvm/Support/Debug.h>
#include <mlir/Transforms/DialectConversion.h>
#include <tuple>
#include <unordered_set>

using namespace mlir;

namespace zkc::Zmir {

mlir::FailureOr<Zmir::ConstructorRefOp> getConstructorRef(func::CallIndirectOp op) {
  auto constrRef = mlir::dyn_cast<Zmir::ConstructorRefOp>(op.getCallee().getDefiningOp());
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
  ) const {
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
    rewriter.replaceOpWithNewOp<NewOp>(op, args);
    return mlir::success();
  }
};

class RemoveConstructorRef : public OpConversionPattern<ConstructorRefOp> {
public:
  using OpConversionPattern<ConstructorRefOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstructorRefOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const {
    if (isLegalConstructRefOp(op)) {
      return mlir::failure();
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

using BitAndPattern = ReplaceConstructorCallWithBuiltIn<Zmir::BitAndOp, BitAndStr>;
using AddPattern = ReplaceConstructorCallWithBuiltIn<Zmir::AddOp, AddStr>;
using SubPattern = ReplaceConstructorCallWithBuiltIn<Zmir::SubOp, SubStr>;
using MulPattern = ReplaceConstructorCallWithBuiltIn<Zmir::MulOp, MulStr>;
using InvPattern = ReplaceConstructorCallWithBuiltIn<Zmir::InvOp, InvStr>;
using IszPattern = ReplaceConstructorCallWithBuiltIn<Zmir::IsZeroOp, IszStr>;
using NegPattern = ReplaceConstructorCallWithBuiltIn<Zmir::NegOp, NegStr>;

namespace {
class LowerBuiltInsPass : public LowerBuiltInsBase<LowerBuiltInsPass> {

  void runOnOperation() override {
    auto op = getOperation();

    Zmir::ZMIRTypeConverter typeConverter;
    mlir::MLIRContext *ctx = op->getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<
        BitAndPattern, AddPattern, SubPattern, MulPattern, InvPattern, IszPattern, NegPattern,
        RemoveConstructorRef>(typeConverter, ctx);

    // Set conversion target
    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<
        zkc::Zmir::ZmirDialect, mlir::func::FuncDialect, mlir::index::IndexDialect,
        mlir::scf::SCFDialect>();
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

} // namespace zkc::Zmir

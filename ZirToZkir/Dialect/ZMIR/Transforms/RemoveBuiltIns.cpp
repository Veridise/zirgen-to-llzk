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

template <const char *Name>
class RemoveBuiltIn : public OpConversionPattern<Zmir::ComponentOp> {
public:
  using OpConversionPattern<Zmir::ComponentOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Zmir::ComponentOp op,
      typename OpConversionPattern<Zmir::ComponentOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto name = op.getName();

    // If it's not the one we are looking for just reject the match
    if (name != Name)
      return mlir::failure();

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

using BitAndPattern = RemoveBuiltIn<BitAndStr>;
using AddPattern = RemoveBuiltIn<AddStr>;
using SubPattern = RemoveBuiltIn<SubStr>;
using MulPattern = RemoveBuiltIn<MulStr>;
using InvPattern = RemoveBuiltIn<InvStr>;
using IszPattern = RemoveBuiltIn<IszStr>;
using NegPattern = RemoveBuiltIn<NegStr>;

namespace {
class RemoveBuiltInsPass : public RemoveBuiltInsBase<RemoveBuiltInsPass> {

  void runOnOperation() override {
    auto op = getOperation();

    Zmir::ZMIRTypeConverter typeConverter;
    mlir::MLIRContext *ctx = op->getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<BitAndPattern, AddPattern, SubPattern, MulPattern, InvPattern,
                 IszPattern, NegPattern>(typeConverter, ctx);

    // Set conversion target
    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<zkc::Zmir::ZmirDialect, mlir::func::FuncDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();

    // Return types may change so we need to adjust the return ops
    target.addDynamicallyLegalOp<Zmir::ComponentOp>([](Zmir::ComponentOp op) {
      return BuiltInComponentNames.find(op.getName().str()) ==
             BuiltInComponentNames.end();
    });

    // Call partialTransformation
    if (mlir::failed(
            mlir::applyFullConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createRemoveBuiltInsPass() {
  return std::make_unique<RemoveBuiltInsPass>();
}

} // namespace zkc::Zmir

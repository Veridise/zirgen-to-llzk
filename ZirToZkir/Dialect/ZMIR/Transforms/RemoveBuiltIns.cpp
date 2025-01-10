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
#include <algorithm>
#include <cassert>
#include <iterator>
#include <llvm/Support/Debug.h>
#include <mlir/Transforms/DialectConversion.h>
#include <tuple>
#include <unordered_set>

using namespace mlir;

namespace zkc::Zmir {

template <const char *Name, typename CompType>
class RemoveBuiltIn : public OpConversionPattern<CompType> {
public:
  using OpConversionPattern<CompType>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CompType op, typename OpConversionPattern<CompType>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter
  ) const override {
    auto name = op.getName();

    // If it's not the one we are looking for just reject the match
    if (name != Name) {
      return mlir::failure();
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

using BitAndPattern = RemoveBuiltIn<BitAndStr, Zmir::ComponentOp>;
using AddPattern = RemoveBuiltIn<AddStr, Zmir::ComponentOp>;
using SubPattern = RemoveBuiltIn<SubStr, Zmir::ComponentOp>;
using MulPattern = RemoveBuiltIn<MulStr, Zmir::ComponentOp>;
using InvPattern = RemoveBuiltIn<InvStr, Zmir::ComponentOp>;
using IszPattern = RemoveBuiltIn<IszStr, Zmir::ComponentOp>;
using NegPattern = RemoveBuiltIn<NegStr, Zmir::ComponentOp>;
using ValPattern = RemoveBuiltIn<ValStr, Zmir::ComponentOp>;
using StringPattern = RemoveBuiltIn<StrStr, Zmir::ComponentOp>;
using ComponentPattern = RemoveBuiltIn<ComponentStr, Zmir::ComponentOp>;
using ArrayPattern = RemoveBuiltIn<ArrayStr, Zmir::ComponentOp>;

namespace {
class RemoveBuiltInsPass : public RemoveBuiltInsBase<RemoveBuiltInsPass> {

  void runOnOperation() override {
    std::unordered_set<std::string> componentsToRemoveSet(
        componentsToRemove.begin(), componentsToRemove.end()
    );

    /*std::transform(*/
    /*    componentsToRemove.begin(), componentsToRemove.end(),*/
    /*    std::inserter(componentsToRemoveSet, componentsToRemoveSet.begin()),*/
    /*    [](auto &s) { return s.str(); }*/
    /*);*/

    auto op = getOperation();

    Zmir::ZMIRTypeConverter typeConverter;
    mlir::MLIRContext *ctx = op->getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<
        BitAndPattern, AddPattern, SubPattern, MulPattern, InvPattern, IszPattern, NegPattern,
        ValPattern, StringPattern, ArrayPattern>(typeConverter, ctx);
    /*fillPatterns<*/
    /*    BitAndStr, AddStr, SubStr, MulStr, InvStr, IszStr, NegStr, ValStr, StringStr,*/
    /*    llzk::StructDefOp>(patterns, typeConverter, ctx);*/

    // Set conversion target
    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<
        zkc::Zmir::ZmirDialect, mlir::func::FuncDialect, index::IndexDialect, scf::SCFDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();

    // Return types may change so we need to adjust the return ops
    target.addDynamicallyLegalOp<Zmir::ComponentOp>([&](Zmir::ComponentOp op) {
      auto found = BuiltInComponentNames.find(op.getName().str()) != BuiltInComponentNames.end();
      auto isTarget =
          componentsToRemoveSet.find(op.getName().str()) != componentsToRemoveSet.end() ||
          componentsToRemoveSet.empty();
      auto markedBuiltIn = op.getBuiltin();
      return !(found && isTarget && markedBuiltIn);
    });

    // Call partialTransformation
    if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createRemoveBuiltInsPass() {
  return std::make_unique<RemoveBuiltInsPass>();
}

} // namespace zkc::Zmir

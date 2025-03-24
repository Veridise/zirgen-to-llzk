// Copyright 2024 Veridise, Inc.

#include <algorithm>
#include <cassert>
#include <iterator>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/DialectConversion.h>
#include <tuple>
#include <unordered_set>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZML/BuiltIns/BuiltIns.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/Types.h>
#include <zklang/Dialect/ZML/Transforms/PassDetail.h>
#include <zklang/Dialect/ZML/Typing/ZMLTypeConverter.h>

using namespace mlir;

namespace zml {

template <const char *Name, typename CompType>
class RemoveBuiltIn : public OpConversionPattern<CompType> {
public:
  using OpConversionPattern<CompType>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CompType op, [[maybe_unused]] typename OpConversionPattern<CompType>::OpAdaptor adaptor,
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

using BitAndPattern = RemoveBuiltIn<BitAndStr, ComponentOp>;
using AddPattern = RemoveBuiltIn<AddStr, ComponentOp>;
using SubPattern = RemoveBuiltIn<SubStr, ComponentOp>;
using MulPattern = RemoveBuiltIn<MulStr, ComponentOp>;
using ExtAddPattern = RemoveBuiltIn<ExtAddStr, ComponentOp>;
using ExtSubPattern = RemoveBuiltIn<ExtSubStr, ComponentOp>;
using ExtMulPattern = RemoveBuiltIn<ExtMulStr, ComponentOp>;
using ModPattern = RemoveBuiltIn<ModStr, ComponentOp>;
using InvPattern = RemoveBuiltIn<InvStr, ComponentOp>;
using ExtInvPattern = RemoveBuiltIn<ExtInvStr, ComponentOp>;
using MakeExtPattern = RemoveBuiltIn<MakeExtStr, ComponentOp>;
using IszPattern = RemoveBuiltIn<IszStr, ComponentOp>;
using NegPattern = RemoveBuiltIn<NegStr, ComponentOp>;
using ValPattern = RemoveBuiltIn<ValStr, ComponentOp>;
using ExtValPattern = RemoveBuiltIn<ExtValStr, ComponentOp>;
using StringPattern = RemoveBuiltIn<StrStr, ComponentOp>;
using ComponentPattern = RemoveBuiltIn<ComponentStr, ComponentOp>;
using ArrayPattern = RemoveBuiltIn<ArrayStr, ComponentOp>;
using InRangePattern = RemoveBuiltIn<InRangeStr, ComponentOp>;

namespace {
class RemoveBuiltInsPass : public RemoveBuiltInsBase<RemoveBuiltInsPass> {

  void runOnOperation() override {
    auto op = getOperation();

    ZMLTypeConverter typeConverter;
    mlir::MLIRContext *ctx = op->getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<
        BitAndPattern, AddPattern, SubPattern, MulPattern, InvPattern, IszPattern, NegPattern,
        ModPattern, ValPattern, StringPattern, ArrayPattern, InRangePattern, ExtAddPattern,
        ExtSubPattern, ExtMulPattern, ExtInvPattern, MakeExtPattern, ExtValPattern>(
        typeConverter, ctx
    );

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<
        ZMLDialect, mlir::func::FuncDialect, index::IndexDialect, scf::SCFDialect,
        arith::ArithDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();

    target.addDynamicallyLegalOp<ComponentOp>([&](ComponentOp op) {
      auto found = BuiltInComponentNames.find(op.getName().str()) != BuiltInComponentNames.end();
      auto markedBuiltIn = op.getBuiltin();
      return !(found && markedBuiltIn);
    });

    if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createRemoveBuiltInsPass() {
  return std::make_unique<RemoveBuiltInsPass>();
}

} // namespace zml

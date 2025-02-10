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
#include "zklang/Dialect/ZML/Typing/ZMIRTypeConverter.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
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
using ModPattern = RemoveBuiltIn<ModStr, Zmir::ComponentOp>;
using InvPattern = RemoveBuiltIn<InvStr, Zmir::ComponentOp>;
using IszPattern = RemoveBuiltIn<IszStr, Zmir::ComponentOp>;
using NegPattern = RemoveBuiltIn<NegStr, Zmir::ComponentOp>;
using ValPattern = RemoveBuiltIn<ValStr, Zmir::ComponentOp>;
using StringPattern = RemoveBuiltIn<StrStr, Zmir::ComponentOp>;
using ComponentPattern = RemoveBuiltIn<ComponentStr, Zmir::ComponentOp>;
using ArrayPattern = RemoveBuiltIn<ArrayStr, Zmir::ComponentOp>;
using InRangePattern = RemoveBuiltIn<InRangeStr, Zmir::ComponentOp>;

namespace {
class RemoveBuiltInsPass : public RemoveBuiltInsBase<RemoveBuiltInsPass> {

  void runOnOperation() override {
    auto op = getOperation();

    Zmir::ZMIRTypeConverter typeConverter;
    mlir::MLIRContext *ctx = op->getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<
        BitAndPattern, AddPattern, SubPattern, MulPattern, InvPattern, IszPattern, NegPattern,
        ModPattern, ValPattern, StringPattern, ArrayPattern, InRangePattern>(typeConverter, ctx);

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<
        zkc::Zmir::ZmirDialect, mlir::func::FuncDialect, index::IndexDialect, scf::SCFDialect,
        arith::ArithDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();

    target.addDynamicallyLegalOp<Zmir::ComponentOp>([&](Zmir::ComponentOp op) {
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

} // namespace zkc::Zmir

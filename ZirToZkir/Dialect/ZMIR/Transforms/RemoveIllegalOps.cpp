
// Copyright 2024 Veridise, Inc.

#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/Transforms/PassDetail.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include <cassert>
#include <llvm/Support/Debug.h>
#include <mlir/Transforms/DialectConversion.h>
#include <tuple>

using namespace mlir;

namespace zkc::Zmir {

namespace {

template <typename Impl, typename Base>
class RemoveIllegalOpsCommon : public Base {

  void runOnOperation() override {
    // Skip if we are not in the interesting function
    if (!inTargetFunction())
      return;

    mlir::RewritePatternSet patterns(&Base::getContext());
    addPatterns(patterns);

    // Set conversion target
    mlir::ConversionTarget target(Base::getContext());
    target.addLegalDialect<zkc::Zmir::ZmirDialect, mlir::func::FuncDialect,
                           zirgen::Zhl::ZhlDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    setLegality(target);

    if (failed(applyPartialConversion(Base::getOperation(), target,
                                      std::move(patterns))))
      Base::signalPassFailure();
  }

protected:
  ComponentInterface getDeclaringComponent() {
    return mlir::dyn_cast<ComponentInterface>(
        Base::getOperation()->getParentOp());
  }

  /// Checks if the function is the one that needs to be transformed
  virtual bool inTargetFunction() = 0;

  /// Set the specific ops that are legal or illegal for the operation.
  virtual void setLegality(ConversionTarget &target) = 0;

  /// Add the patterns
  virtual void addPatterns(RewritePatternSet &patterns) = 0;
};

template <typename Op> class RemoveOp : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename OpConversionPattern<Op>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

template <typename Op, int ArgIdx>
class ReplaceUsesWithArg : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename OpConversionPattern<Op>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto body = op->template getParentOfType<mlir::func::FuncOp>();
    mlir::BlockArgument arg = body.getArgument(ArgIdx);

    rewriter.replaceAllUsesWith(op, arg);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ReplaceConstructWithRead
    : public OpConversionPattern<func::CallIndirectOp> {
public:
  using OpConversionPattern<func::CallIndirectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallIndirectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    return mlir::failure();
  }
};

class ReplaceWriteFieldWithRead : public OpConversionPattern<WriteFieldOp> {
public:
  using OpConversionPattern<WriteFieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WriteFieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    if (auto valOp = adaptor.getVal().getDefiningOp()) {
      llvm::dbgs() << *valOp << "\n";
      rewriter.setInsertionPoint(valOp);
    }
    auto read = rewriter.create<Zmir::ReadFieldOp>(
        op.getLoc(), op.getVal().getType(), adaptor.getComponent(),
        adaptor.getFieldNameAttr());
    rewriter.eraseOp(op);
    rewriter.replaceUsesWithIf(adaptor.getVal(), read, [](auto &operand) {
      // Replace anything but write ops since we want to get rid of them
      return !mlir::isa<WriteFieldOp>(operand.getOwner());
    });
    return mlir::success();
  }
};

class RemoveIllegalComputeOpsPass
    : public RemoveIllegalOpsCommon<
          RemoveIllegalComputeOpsPass,
          RemoveIllegalComputeOpsBase<RemoveIllegalComputeOpsPass>> {

  bool inTargetFunction() override {
    auto comp = getDeclaringComponent();
    if (!comp || comp.hasUnifiedBody())
      return false;
    return getOperation() == comp.getBodyFunc();
  }

  void setLegality(ConversionTarget &target) override {
    target.addIllegalOp<ConstrainOp>();
  }

  void addPatterns(RewritePatternSet &patterns) override {
    patterns.add<RemoveOp<ConstrainOp>>(&getContext());
  }
};

class RemoveIllegalConstrainOpsPass
    : public RemoveIllegalOpsCommon<
          RemoveIllegalConstrainOpsPass,
          RemoveIllegalConstrainOpsBase<RemoveIllegalConstrainOpsPass>> {

  bool inTargetFunction() override {
    auto comp = getDeclaringComponent();
    if (!comp || comp.hasUnifiedBody())
      return false;
    return getOperation() == comp.getConstrainFunc();
  }

  void setLegality(ConversionTarget &target) override {
    // And there's probably more
    target.addIllegalOp<WriteFieldOp, GetSelfOp, BitAndOp, InvOp>();
    target.addIllegalOp<func::CallIndirectOp>();
  }

  void addPatterns(RewritePatternSet &patterns) override {
    patterns
        .add<ReplaceWriteFieldWithRead, RemoveOp<BitAndOp>, RemoveOp<InvOp>,
             RemoveOp<func::CallIndirectOp>, ReplaceUsesWithArg<GetSelfOp, 0>>(
            &getContext());
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createRemoveIllegalComputeOpsPass() {
  return std::make_unique<RemoveIllegalComputeOpsPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
createRemoveIllegalConstrainOpsPass() {
  return std::make_unique<RemoveIllegalConstrainOpsPass>();
}

} // namespace zkc::Zmir

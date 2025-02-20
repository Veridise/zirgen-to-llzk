
// Copyright 2024 Veridise, Inc.

#include <cassert>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <tuple>
#include <type_traits>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/Transforms/PassDetail.h>
#include <zklang/Dialect/ZML/Utils/Patterns.h>

using namespace mlir;

namespace zml {

namespace {

template <typename Impl, typename Base> class RemoveIllegalOpsCommon : public Base {

  void runOnOperation() override {
    // Skip if we are not in the interesting function
    if (!inTargetFunction()) {
      return;
    }

    mlir::RewritePatternSet patterns(&Base::getContext());
    addPatterns(patterns);

    // Set conversion target
    mlir::ConversionTarget target(Base::getContext());
    target.addLegalDialect<
        ZMLDialect, mlir::func::FuncDialect, zirgen::Zhl::ZhlDialect, mlir::index::IndexDialect,
        mlir::scf::SCFDialect, arith::ArithDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    setLegality(target);

    if (failed(applyPartialConversion(Base::getOperation(), target, std::move(patterns)))) {
      Base::signalPassFailure();
    }
  }

protected:
  ComponentInterface getDeclaringComponent() {
    return mlir::dyn_cast<ComponentInterface>(Base::getOperation()->getParentOp());
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

  LogicalResult matchAndRewrite(
      Op op, typename OpConversionPattern<Op>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter
  ) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

template <typename Op, int ArgIdx> class ReplaceUsesWithArg : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Op op, typename OpConversionPattern<Op>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter
  ) const override {
    auto body = op->template getParentOfType<mlir::func::FuncOp>();
    mlir::BlockArgument arg = body.getArgument(ArgIdx);

    rewriter.replaceAllUsesWith(op, arg);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ReplaceConstructWithRead : public OpConversionPattern<func::CallIndirectOp> {
public:
  using OpConversionPattern<func::CallIndirectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::CallIndirectOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {

    return mlir::failure();
  }
};

class ReplaceWriteFieldWithRead : public OpConversionPattern<WriteFieldOp> {
public:
  using OpConversionPattern<WriteFieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      WriteFieldOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {
    insertReadToReplaceUses(op, adaptor, rewriter);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  void insertReadToReplaceUses(
      WriteFieldOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const {

    OpBuilder::InsertionGuard guard(rewriter);
    /// Only insert the read op if the value is primitive
    if (mlir::isa<ComponentType>(op.getVal().getType())) {
      return;
    }

    if (auto valOp = adaptor.getVal().getDefiningOp()) {
      rewriter.setInsertionPoint(valOp);
    } else {
      rewriter.setInsertionPointToStart(&(op->getParentOfType<mlir::func::FuncOp>().getBody().front(
      )));
    }
    auto read = rewriter.create<ReadFieldOp>(
        op.getLoc(), op.getVal().getType(), adaptor.getComponent(), adaptor.getFieldNameAttr()
    );
    rewriter.replaceUsesWithIf(adaptor.getVal(), read, [](auto &operand) {
      // Replace anything but write ops since we want to get rid of them
      // and changing these ops causes recursion problems with the pattern
      // matcher.
      return !mlir::isa<WriteFieldOp>(operand.getOwner());
    });
  }
};

class ReplaceWriteArrayWithRead : public OpConversionPattern<WriteArrayOp> {
public:
  using OpConversionPattern<WriteArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      WriteArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
  ) const override {

    return success();
  }
};

class RemoveIllegalComputeOpsPass
    : public RemoveIllegalOpsCommon<
          RemoveIllegalComputeOpsPass, RemoveIllegalComputeOpsBase<RemoveIllegalComputeOpsPass>> {

  bool inTargetFunction() override {
    auto comp = getDeclaringComponent();
    if (!comp || comp.hasUnifiedBody()) {
      return false;
    }
    return getOperation() == comp.getBodyFunc();
  }

  void setLegality(ConversionTarget &target) override {
    target.addIllegalOp<ConstrainOp, ConstrainCallOp>();
  }

  void addPatterns(RewritePatternSet &patterns) override {
    patterns.add<RemoveOp<ConstrainOp>, RemoveOp<ConstrainCallOp>>(&getContext());
  }
};

class RemoveIllegalConstrainOpsPass
    : public RemoveIllegalOpsCommon<
          RemoveIllegalConstrainOpsPass,
          RemoveIllegalConstrainOpsBase<RemoveIllegalConstrainOpsPass>> {

  bool inTargetFunction() override {
    auto comp = getDeclaringComponent();
    if (!comp || comp.hasUnifiedBody()) {
      return false;
    }
    return getOperation() == comp.getConstrainFunc();
  }

  void setLegality(ConversionTarget &target) override {
    target.addIllegalOp<
        WriteFieldOp, ConstructorRefOp,
        SelfOp, // Gets transformed into llzk.new_struct
        BitAndOp, InvOp>();
    target.addDynamicallyLegalOp<func::CallIndirectOp>([](func::CallIndirectOp callOp) {
      auto calleeOp = callOp.getCallee().getDefiningOp();
      if (!calleeOp) {
        return false;
      }
      return mlir::isa<ExternFnRefOp>(calleeOp);
    }); // Gets transformed into a call to @compute
    target.addDynamicallyLegalOp<WriteArrayOp>([](WriteArrayOp writeArrOp) {
      return !writeArrOp.getComputeOnly();
    });
  }

  void addPatterns(RewritePatternSet &patterns) override {
    patterns.add<
        RemoveOp<WriteFieldOp>, RemoveOp<BitAndOp>, RemoveOp<InvOp>, ReplaceSelfWith<Arg<0>>,
        RemoveOp<WriteArrayOp>, RemoveOp<func::CallIndirectOp>, RemoveOp<ConstructorRefOp>>(
        &getContext()
    );
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createRemoveIllegalComputeOpsPass() {
  return std::make_unique<RemoveIllegalComputeOpsPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createRemoveIllegalConstrainOpsPass() {
  return std::make_unique<RemoveIllegalConstrainOpsPass>();
}

} // namespace zml

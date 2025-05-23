//===- RemoveIllegalOps.cpp - Illegal ops removal ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the implementation of the --remove-illegal-compute-ops
// and --remove-illegal-constrain-ops passes.
//
//===----------------------------------------------------------------------===//

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

template <typename Base> class RemoveIllegalOpsCommon : public Base {

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
      Op op, typename OpConversionPattern<Op>::OpAdaptor, ConversionPatternRewriter &rewriter
  ) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class RemoveIllegalComputeOpsPass
    : public RemoveIllegalOpsCommon<RemoveIllegalComputeOpsBase<RemoveIllegalComputeOpsPass>> {

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
    : public RemoveIllegalOpsCommon<RemoveIllegalConstrainOpsBase<RemoveIllegalConstrainOpsPass>> {

  bool inTargetFunction() override {
    auto comp = getDeclaringComponent();
    if (!comp || comp.hasUnifiedBody()) {
      return false;
    }
    return getOperation() == comp.getConstrainFunc();
  }

  void setLegality(ConversionTarget &target) override {
    target.addIllegalOp<
        WriteFieldOp, SetGlobalOp, ConstructorRefOp,
        SelfOp, // Gets transformed into llzk.new_struct
        BitAndOp, InvOp, ExtInvOp, EqzExtOp>();
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
        RemoveOp<WriteFieldOp>, RemoveOp<SetGlobalOp>, RemoveOp<BitAndOp>, RemoveOp<InvOp>,
        RemoveOp<ExtInvOp>, RemoveOp<EqzExtOp>, ReplaceSelfWith<Arg<0>>, RemoveOp<WriteArrayOp>,
        RemoveOp<func::CallIndirectOp>, RemoveOp<ConstructorRefOp>>(&getContext());
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

//===- SplitComponentBody.cpp - Witness & constraints splitting -*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the implementation of the --split-component-body pass.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Transforms/DialectConversion.h>
#include <tuple>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/Transforms/PassDetail.h>

using namespace mlir;

namespace zml {

namespace {

class PendingSymbolRenames {
public:
  explicit PendingSymbolRenames(ModuleOp);

  bool hasPendingRenames() const;
  void removeOp(Operation *);
  void addOp(Operation *, StringAttr);
  LogicalResult applyPendingRenames();

private:
  SymbolTable st;
  // Maps the autogenerated name to the desired symbol name
  std::vector<std::tuple<mlir::Operation *, mlir::StringAttr, mlir::StringAttr>> pending;
};

PendingSymbolRenames::PendingSymbolRenames(ModuleOp mod) : st(mod) {}

bool PendingSymbolRenames::hasPendingRenames() const { return !pending.empty(); }

/// Removes the op from the symbol table and from the pendings list if found
void PendingSymbolRenames::removeOp(Operation *op) {
  st.remove(op); // Only remove from the symbol table
  auto toRemove =
      std::remove_if(pending.begin(), pending.end(), [&](auto t) { return std::get<0>(t) == op; });
  pending.erase(toRemove, pending.end());
}

/// Inserts the operation in the symbol table and marks it as
/// pending with the desired name.
void PendingSymbolRenames::addOp(Operation *op, StringAttr desired) {
  pending.push_back({op, st.insert(op), desired});
}

/// Applies the pending rename operations.
/// The queue of pending operations gets cleared regardless
/// of the result of the rename operation.
LogicalResult PendingSymbolRenames::applyPendingRenames() {
  std::vector<LogicalResult> results;
  std::transform(pending.begin(), pending.end(), std::back_inserter(results), [&](auto rename) {
    // Give them more descriptive names
    auto *op = std::get<0>(rename);
    auto &assigned = std::get<1>(rename);
    auto &desired = std::get<2>(rename);

    // No need to rename if they are already equal
    if (assigned == desired) {
      return mlir::success();
    }

    return st.rename(op, desired);
  });
  pending.clear();

  return mlir::success(std::all_of(results.begin(), results.end(), mlir::succeeded));
}

class SplitComponentOpPattern : public mlir::OpConversionPattern<ComponentOp> {
public:
  template <typename... Args>
  SplitComponentOpPattern(PendingSymbolRenames &Pending, Args &&...args)
      : ::OpConversionPattern<ComponentOp>(std::forward<Args>(args)...), pending(Pending) {}

  mlir::LogicalResult
  matchAndRewrite(ComponentOp, OpAdaptor, mlir::ConversionPatternRewriter &) const override;

private:
  PendingSymbolRenames &pending;
};

class ReplaceReturnOpInConstrainFunc : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
public:
  using OpConversionPattern<mlir::func::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(func::ReturnOp, OpAdaptor, ConversionPatternRewriter &) const override;
};

LogicalResult SplitComponentOpPattern::matchAndRewrite(
    ComponentOp op, OpAdaptor, mlir::ConversionPatternRewriter &rewriter
) const {

  SplitComponentOp newOp = rewriter.create<SplitComponentOp>(
      op.getLoc(), op->getResultTypes(), op->getOperands(), op->getAttrs()
  );

  // Make the symbol table aware of the new component op
  // but don't name it like the old op right now.
  pending.removeOp(op);
  pending.addOp(newOp, op.getSymNameAttr());

  mlir::OpBuilder::InsertionGuard insertionGuard(rewriter);
  auto *block = rewriter.createBlock(&newOp.getRegion());
  rewriter.setInsertionPointToStart(block);

  // Copy the field definitions
  for (auto paramOp : op.getOps<FieldDefOp>()) {
    rewriter.clone(*paramOp.getOperation());
  }
  // Copy additional functions declared (like externs)
  for (auto f : op.getOps<mlir::func::FuncOp>()) {
    if (f == op.getBodyFunc()) {
      continue;
    }
    rewriter.clone(*f.getOperation());
  }

  auto bodyFuncType = op.getBodyFunc().getFunctionType();

  mlir::IRMapping computeMapping;
  auto computeFunc = mlir::cast<mlir::func::FuncOp>(
      rewriter.clone(*op.getBodyFunc().getOperation(), computeMapping)
  );
  computeFunc.setName(newOp.getBodyFuncName());

  std::vector<mlir::Type> constrainFuncArgTypes({op.getType()});
  constrainFuncArgTypes.insert(
      constrainFuncArgTypes.end(), bodyFuncType.getInputs().begin(), bodyFuncType.getInputs().end()
  );
  auto constrainFuncType = rewriter.getFunctionType(constrainFuncArgTypes, TypeRange());
  mlir::IRMapping contrainMapping;
  auto constrainFunc = mlir::cast<mlir::func::FuncOp>(
      rewriter.clone(*op.getBodyFunc().getOperation(), contrainMapping)
  );
  constrainFunc.setName(newOp.getConstrainFuncName());
  constrainFunc.setFunctionType(constrainFuncType);

  // Insert the self argument for the constrain function as it will be
  // required by llzk.
  assert(!constrainFunc.getRegion().empty() && "was expecting a filled out function");

  auto &b = constrainFunc.getRegion().front();
  if (b.args_empty()) {
    b.addArgument(op.getType(), op.getLoc());
  } else {
    b.insertArgument(b.args_begin(), op.getType(), op.getLoc());
  }

  rewriter.eraseOp(op);

  return mlir::success();
}

LogicalResult ReplaceReturnOpInConstrainFunc::matchAndRewrite(
    func::ReturnOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  auto func = op.getParentOp();
  auto comp = mlir::dyn_cast<ComponentInterface>(func->getParentOp());
  if (!comp || (func != comp.getConstrainFunc() && !comp.hasUnifiedBody())) {
    return mlir::failure();
  }

  // At this point we know we are in a return op of a constrain function, so we
  // need to make a void return since that's the return type of that function.
  rewriter.replaceOpWithNewOp<func::ReturnOp>(op, TypeRange(), ValueRange());

  return mlir::success();
}

class SplitComponentBodyPass : public SplitComponentBodyBase<SplitComponentBodyPass> {

  void runOnOperation() override {
    auto op = getOperation();
    PendingSymbolRenames pending(op);

    mlir::MLIRContext *ctx = op->getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<SplitComponentOpPattern>(pending, ctx);
    patterns.add<ReplaceReturnOpInConstrainFunc>(ctx);

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<
        ZMLDialect, mlir::func::FuncDialect, index::IndexDialect, scf::SCFDialect,
        arith::ArithDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp, mlir::ModuleOp>();
    target.addIllegalDialect<zirgen::Zhl::ZhlDialect>();
    target.addIllegalOp<ComponentOp>();

    target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp ret) {
      return ret.getOperandTypes() ==
             ret.getParentOp<func::FuncOp>().getFunctionType().getResults();
    });

    if (mlir::failed(mlir::applyFullConversion(op, target, std::move(patterns))) ||
        // If the transformation went OK then we try to apply the pending
        // renames if any
        (pending.hasPendingRenames() && mlir::failed(pending.applyPendingRenames()))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createSplitComponentBodyPass() {
  return std::make_unique<SplitComponentBodyPass>();
}

} // namespace zml

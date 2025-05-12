//===- TypecheckZhl.cpp - zirgen type checking pass -------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the implementation for the --typecheck-zhl pass.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZHL/Typing/Analysis.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Passes/PassDetail.h>
#include <zklang/Passes/Passes.h>

using namespace mlir;
using namespace zklang;
using namespace zhl;
using namespace zml;

namespace {

class TypecheckZhlPass : public TypecheckZhlBase<TypecheckZhlPass> {

  void runOnOperation() override {
    auto &analysis = getAnalysis<ZIRTypeAnalysis>();
    markAnalysesPreserved<ZIRTypeAnalysis>();
    auto *ctx = &getContext();
    Builder builder(ctx);

    for (auto &[value, binding] : analysis.exprs()) {
      if (failed(binding)) {
        continue; // Don't inject on failing checks
      }
      StringAttr key = nullptr;
      Operation *op = nullptr;
      if (auto result = llvm::dyn_cast<OpResult>(value)) {
        op = result.getDefiningOp();
        key = builder.getStringAttr("zml.binding");
      }
      if (auto arg = llvm::dyn_cast<BlockArgument>(value)) {
        auto block = arg.getOwner();
        assert(block->isEntryBlock() && "only entry blocks supported for annotation");
        op = block->getParentOp();
        key = builder.getStringAttr("zml.arg_binding." + Twine(arg.getArgNumber()));
      }
      if (op) {
        assert(key);
        llvm::errs() << "operation ";
        op->print(llvm::errs());
        llvm::errs() << " will have binding " << *binding << "\n";
        auto attr = FixedTypeBindingAttr::get(ctx, *binding, analysis.getBindings());
        llvm::errs() << "   Converts to attr " << attr << "\n";
        op->setDiscardableAttr(key, attr);
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> zklang::createTypecheckZhlPass() {
  return std::make_unique<TypecheckZhlPass>();
}

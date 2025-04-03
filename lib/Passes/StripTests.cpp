//===- StripTests.cpp - zirgen tests removal --------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the implementation for the --strip-tests pass.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Passes/PassDetail.h>

using namespace mlir;

namespace zklang {

namespace {

class StripTestsPass : public StripTestsBase<StripTestsPass> {

  void runOnOperation() override {
    DenseSet<Operation *> toErase;
    for (auto &op : getOperation().getOps()) {
      auto compOp = llvm::dyn_cast<zirgen::Zhl::ComponentOp>(op);
      if (!compOp) {
        continue;
      }
      auto symName = compOp.getName();
      if (symName.starts_with("test$") || symName.contains("$test")) {
        toErase.insert(&op);
      }
    }

    for (auto op : toErase) {
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createStripTestsPass() {
  return std::make_unique<StripTestsPass>();
}

} // namespace zklang

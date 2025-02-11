// Copyright 2024 Veridise, Inc.

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Passes/PassDetail.h"
#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <mlir/Pass/Pass.h>

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

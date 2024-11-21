// Copyright 2024 Veridise, Inc.

#include "ZirToZkir/Passes/PassDetail.h"
#include "mlir/IR/BuiltinOps.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;

namespace zkc {

namespace {

class StripDirectivesPass : public StripDirectivesBase<StripDirectivesPass> {

  void runOnOperation() override {
    DenseSet<zirgen::Zhl::DirectiveOp *> toErase;

    for (auto &op : getOperation().getOps()) {
      auto compOp = llvm::dyn_cast<zirgen::Zhl::ComponentOp>(op);
      if (!compOp)
        continue;
      for (auto dirOp :
           compOp.getRegion().front().getOps<zirgen::Zhl::DirectiveOp>()) {
        toErase.insert(&dirOp);
      }
    }

    for (auto op : toErase)
      op->erase();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createStripDirectivesPass() {
  return std::make_unique<StripDirectivesPass>();
}

} // namespace zkc

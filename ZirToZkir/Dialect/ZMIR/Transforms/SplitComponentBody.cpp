
// Copyright 2024 Veridise, Inc.

#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/Transforms/PassDetail.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include <cassert>
#include <llvm/Support/Debug.h>

using namespace mlir;

namespace zkc::Zmir {

namespace {

class SplitComponentBodyPass
    : public SplitComponentBodyBase<SplitComponentBodyPass> {

  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<OperationPass<ComponentOp>> createInjectBuiltInsPass() {

  return std::make_unique<SplitComponentBodyPass>();
}

} // namespace zkc::Zmir

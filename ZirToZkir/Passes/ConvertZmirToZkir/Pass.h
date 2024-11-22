#pragma once

// Copyright 2024 Veridise, Inc.

#include "ZirToZkir/Passes/PassDetail.h"
#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>

namespace zkc {

class ConvertZmirToZkirPass
    : public ConvertZmirToZkirBase<ConvertZmirToZkirPass> {
  void runOnOperation() override;
};

} // namespace zkc

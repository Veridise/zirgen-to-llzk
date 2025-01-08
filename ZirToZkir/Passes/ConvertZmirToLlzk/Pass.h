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

class ConvertZmirToLlzkPass : public ConvertZmirToLlzkBase<ConvertZmirToLlzkPass> {
  void runOnOperation() override;
};

class ConvertZmirComponentsToLlzkPass
    : public ConvertZmirComponentsToLlzkBase<ConvertZmirComponentsToLlzkPass> {
  void runOnOperation() override;
};

} // namespace zkc

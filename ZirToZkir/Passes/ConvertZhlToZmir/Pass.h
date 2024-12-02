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

class ConvertZhlToZmirPass : public ConvertZhlToZmirBase<ConvertZhlToZmirPass> {
  void runOnOperation() override;
};

class TransformComponentDeclsPass
    : public TransformComponentDeclsBase<TransformComponentDeclsPass> {
  void runOnOperation() override;
};

class ConvertZhlToScfPass : public ConvertZhlToScfBase<ConvertZhlToScfPass> {
  void runOnOperation() override;
};

} // namespace zkc

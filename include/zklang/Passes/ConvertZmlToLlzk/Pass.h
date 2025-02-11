#pragma once

// Copyright 2024 Veridise, Inc.

#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Passes/PassDetail.h>

namespace zml {

class ConvertZmlToLlzkPass : public zklang::ConvertZmlToLlzkBase<ConvertZmlToLlzkPass> {
  void runOnOperation() override;
};

class InjectLlzkModAttrsPass : public zklang::InjectLlzkModAttrsBase<InjectLlzkModAttrsPass> {
  void runOnOperation() override;
};

} // namespace zml

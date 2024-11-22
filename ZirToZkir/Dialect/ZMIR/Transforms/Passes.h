// Copyright 2024 Veridise, Inc.

#pragma once

#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace zkc::Zmir {

// Pass constructors
template <typename Op> using OpPass = std::unique_ptr<mlir::OperationPass<Op>>;

OpPass<mlir::ModuleOp> createInjectBuiltInsPass();
// OpPass<ComponentOp> createSplitComponentBodyPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "ZirToZkir/Dialect/ZMIR/Transforms/Passes.inc.h"

} // namespace zkc::Zmir

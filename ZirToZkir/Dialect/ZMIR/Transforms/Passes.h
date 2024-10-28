// Copyright 2024 Veridise, Inc.

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace zkc::Zmir {

// Pass constructors
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createInjectBuiltInsPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "ZirToZkir/Dialect/ZMIR/Transforms/Passes.h.inc"

} // namespace zkc::Zmir

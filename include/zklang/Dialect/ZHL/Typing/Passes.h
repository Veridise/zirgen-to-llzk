// Copyright 2024 Veridise, Inc.

#pragma once

/*#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"*/
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace zhl {

// Pass constructors
template <typename Op> using OpPass = std::unique_ptr<mlir::OperationPass<Op>>;

OpPass<mlir::ModuleOp> createPrintTypeBindingsPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "ZirToZkir/Dialect/ZHL/Typing/Passes.inc.h"

} // namespace zhl

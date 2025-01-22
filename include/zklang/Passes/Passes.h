// Copyright 2024 Veridise, Inc.

#pragma once

#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "zklang/Dialect/ZML/IR/Dialect.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace zkc {

// Pass constructors

template <typename Op> using OpPass = std::unique_ptr<mlir::OperationPass<Op>>;

OpPass<mlir::ModuleOp> createStripTestsPass();
OpPass<mlir::ModuleOp> createStripDirectivesPass();
OpPass<mlir::ModuleOp> createConvertZhlToZmirPass();
OpPass<llzk::StructDefOp> createConvertZmirToLlzkPass();
OpPass<mlir::ModuleOp> createConvertZmirComponentsToLlzkPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "zklang/Passes/Passes.h.inc"

} // namespace zkc

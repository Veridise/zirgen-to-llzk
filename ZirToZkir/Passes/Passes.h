// Copyright 2024 Veridise, Inc.

#pragma once

#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "zkir/Dialect/ZKIR/IR/Dialect.h"
#include "zkir/Dialect/ZKIR/IR/Ops.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace zkc {

// Pass constructors

template <typename Op> using OpPass = std::unique_ptr<mlir::OperationPass<Op>>;

OpPass<mlir::ModuleOp> createStripTestsPass();
OpPass<mlir::ModuleOp> createStripDirectivesPass();
OpPass<mlir::ModuleOp> createConvertZhlToZmirPass();
OpPass<zkir::StructDefOp> createConvertZmirToZkirPass();
OpPass<mlir::ModuleOp> createConvertZmirComponentsToZkirPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "ZirToZkir/Passes/Passes.inc.h"

} // namespace zkc

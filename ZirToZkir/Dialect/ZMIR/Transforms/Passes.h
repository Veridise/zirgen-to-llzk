// Copyright 2024 Veridise, Inc.

#pragma once

#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace zkc::Zmir {

// Pass constructors
template <typename Op> using OpPass = std::unique_ptr<mlir::OperationPass<Op>>;

OpPass<mlir::ModuleOp> createInjectBuiltInsPass();
OpPass<mlir::ModuleOp> createRemoveBuiltInsPass();
OpPass<mlir::ModuleOp> createSplitComponentBodyPass();
OpPass<mlir::func::FuncOp> createLowerBuiltInsPass();
OpPass<mlir::func::FuncOp> createRemoveIllegalComputeOpsPass();
OpPass<mlir::func::FuncOp> createRemoveIllegalConstrainOpsPass();
OpPass<ComponentOp> createInsertTemporariesPass();
OpPass<ComponentOp> createLegalizeTypesPass();
OpPass<ComponentOp> createExpandSuperCoercionPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "ZirToZkir/Dialect/ZMIR/Transforms/Passes.inc.h"

} // namespace zkc::Zmir

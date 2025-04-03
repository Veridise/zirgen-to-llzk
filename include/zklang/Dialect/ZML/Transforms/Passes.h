//===- Passes.h - Passes public API ---------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the factory methods and registration functions of the
// passes defined in Passes.td in the same directory.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <zklang/Dialect/ZML/IR/Ops.h>

namespace zml {

// Pass constructors
template <typename Op> using OpPass = std::unique_ptr<mlir::OperationPass<Op>>;

OpPass<mlir::ModuleOp> createInjectBuiltInsPass();
OpPass<mlir::ModuleOp> createRemoveBuiltInsPass();
OpPass<mlir::ModuleOp> createSplitComponentBodyPass();
OpPass<mlir::func::FuncOp> createLowerBuiltInsPass();
OpPass<mlir::func::FuncOp> createRemoveIllegalComputeOpsPass();
OpPass<mlir::func::FuncOp> createRemoveIllegalConstrainOpsPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <zklang/Dialect/ZML/Transforms/Passes.h.inc>

} // namespace zml

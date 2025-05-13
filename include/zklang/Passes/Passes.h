//===- Passes.h - Passes public API -----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the factory and registration functions for the passes
// defined in Passes.td in the same directory.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llzk/Dialect/Function/IR/Dialect.h>
#include <llzk/Dialect/LLZK/IR/Dialect.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/IR/Ops.h>

namespace zklang {

// Pass constructors

template <typename Op> using OpPass = std::unique_ptr<mlir::OperationPass<Op>>;

OpPass<mlir::ModuleOp> createStripTestsPass();
OpPass<mlir::ModuleOp> createAnnotateTypecheckZhlPass();
OpPass<mlir::ModuleOp> createConvertZhlToLlzkStructPass();
OpPass<mlir::ModuleOp> createConvertZhlToZmlPass();
OpPass<mlir::ModuleOp> createConvertZmlToLlzkPass();
OpPass<mlir::ModuleOp> createInjectLlzkModAttrsPass();
OpPass<mlir::ModuleOp> createInstantiatePODBlocksPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <zklang/Passes/Passes.h.inc>

} // namespace zklang

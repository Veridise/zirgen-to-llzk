//===- Passes.h - Passes public API -----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the factory methods and pass registration for the passes
// defined in Passes.td in the same directory.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace zhl {

// Pass constructors
template <typename Op> using OpPass = std::unique_ptr<mlir::OperationPass<Op>>;

OpPass<mlir::ModuleOp> createPrintTypeBindingsPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <zklang/Dialect/ZHL/Typing/Passes.h.inc>

} // namespace zhl

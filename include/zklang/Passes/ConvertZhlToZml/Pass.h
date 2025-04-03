//===- Pass.h - ZHL->ZML conversion pass ------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the pass that converts ZHL operations into ZML operations.
//
//===----------------------------------------------------------------------===//

#pragma once

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

class ConvertZhlToZmlPass : public zklang::ConvertZhlToZmlBase<ConvertZhlToZmlPass> {
  void runOnOperation() override;
};

} // namespace zml

//===- Pass.h - ZML->LLZK conversion pass -----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the pass that converts ZML operations into LLZK
// operations and the pass that adds the necessary attributes to the top-level
// LLZK module.
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

class ConvertZmlToLlzkPass : public zklang::ConvertZmlToLlzkBase<ConvertZmlToLlzkPass> {
  void runOnOperation() override;
};

class InjectLlzkModAttrsPass : public zklang::InjectLlzkModAttrsBase<InjectLlzkModAttrsPass> {
  void runOnOperation() override;
};

} // namespace zml

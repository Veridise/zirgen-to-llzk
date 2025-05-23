//===- PrintTypeBindings.cpp - Pass to dump type bindings -------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the implementation of the --zhl-print-type-bindings pass.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <numeric>
#include <zklang/Dialect/ZHL/Typing/Analysis.h>
#include <zklang/Dialect/ZHL/Typing/OpBindings.h>
#include <zklang/Dialect/ZHL/Typing/PassDetail.h>
#include <zklang/Dialect/ZHL/Typing/Scope.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>
#include <zklang/Dialect/ZHL/Typing/Typing.h>
#include <zklang/Dialect/ZML/BuiltIns/BuiltIns.h>

using namespace zirgen::Zhl;
using namespace mlir;

namespace zhl {

namespace {

class PrintTypeBindingsPass : public PrintTypeBindingsBase<PrintTypeBindingsPass> {

  void runOnOperation() override {
    auto &analysis = getAnalysis<ZIRTypeAnalysis>();
    if (mlir::failed(analysis)) {
      signalPassFailure();
      return;
    }
    analysis.emitRemarks();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createPrintTypeBindingsPass() {
  return std::make_unique<PrintTypeBindingsPass>();
}

} // namespace zhl

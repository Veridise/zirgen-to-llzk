//===- AnnotateTypecheckZhl.cpp - zirgen type checking pass -----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the implementation for the --annotate-typecheck-zhl pass.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZHL/Typing/Analysis.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/Typing/Annotations.h>
#include <zklang/Passes/PassDetail.h>
#include <zklang/Passes/Passes.h>

using namespace mlir;
using namespace zklang;
using namespace zhl;
using namespace zml;

namespace {

class AnnotateTypecheckZhlPass : public AnnotateTypecheckZhlBase<AnnotateTypecheckZhlPass> {

  void runOnOperation() override {
    auto &analysis = getAnalysis<ZIRTypeAnalysis>();
    markAnalysesPreserved<ZIRTypeAnalysis>();
    auto *ctx = &getContext();
    annotateOperations(ctx, getOperation(), analysis);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> zklang::createAnnotateTypecheckZhlPass() {
  return std::make_unique<AnnotateTypecheckZhlPass>();
}

// REMOVE ME!!!! NOT NEEDED!!!

//===- InstantiatePODBlocks.cpp - pod blocks instantiation ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the implementation for the --instantiate-pod-blocks pass.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZHL/Typing/Analysis.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZML/ExtVal/Conversion.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/Typing/Annotations.h>
#include <zklang/Dialect/ZML/Utils/Helpers.h>
#include <zklang/Passes/PassDetail.h>
#include <zklang/Passes/Passes.h>

using namespace mlir;
using namespace zklang;
using namespace zhl;
using namespace zml;

static void createPODComponentsFromClosures(
    zhl::ZIRTypeAnalysis &analysis, OpBuilder &builder, SymbolTable &st, Block *insertionPoint,
    const mlir::TypeConverter &tc
) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(insertionPoint);
  for (auto *clo : analysis.getClosures()) {
    assert(clo && clo->hasClosure());
    createPODComponent(*clo, builder, st, tc);
  }
}

namespace {

class InstantiatePODBlocksPass : public InstantiatePODBlocksBase<InstantiatePODBlocksPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto conversion =
        extval::loadConversionByFieldName(selectedExtValField, [&mod] { return mod->emitError(); });
    // Only BabyBear is supported for now. More to come (LLZK-180)
    if (failed(conversion)) {
      llvm::errs() << "Failed to configure field\n";
      signalPassFailure();
      return;
    }
    auto &analysis = getAnalysis<ZIRTypeAnalysis>();
    markAnalysesPreserved<ZIRTypeAnalysis>();
    OpBuilder builder(mod);
    SymbolTable st(mod);
    createPODComponentsFromClosures(
        analysis, builder, st, &mod.getRegion().front(), *conversion->typeConverter
    );
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> zklang::createInstantiatePODBlocksPass() {
  return std::make_unique<InstantiatePODBlocksPass>();
}

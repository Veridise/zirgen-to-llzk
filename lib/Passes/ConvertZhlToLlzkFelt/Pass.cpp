//===- Pass.cpp - zhl to llzk felt pass -------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the implementation for the
// --convert-zhl-to-llzk-felt pass.
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
#include <zklang/Dialect/ZML/ExtVal/Conversion.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/Typing/Annotations.h>
#include <zklang/Passes/ConvertZhlToLlzkFelt/Patterns.h>
#include <zklang/Passes/PassDetail.h>
#include <zklang/Passes/Passes.h>

using namespace mlir;
using namespace zklang;
using namespace zhl;
using namespace zml;
using namespace llzk::component;

namespace {

class ConvertZhlToLlzkFeltPass : public ConvertZhlToLlzkFeltBase<ConvertZhlToLlzkFeltPass> {

  void runOnOperation() override {

    auto op = getOperation();

    MLIRContext *ctx = &getContext();
    auto conversion =
        extval::loadConversionByFieldName(selectedExtValField, [&op] { return op->emitError(); });
    // Only BabyBear is supported for now. More to come (LLZK-180)
    if (failed(conversion)) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    populateZhlToLlzkFeltConversionPatternsAndLegality(
        *conversion->typeConverter, patterns, target
    );

    if (failed(applyFullConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<StructDefOp>> zklang::createConvertZhlToLlzkFeltPass() {
  return std::make_unique<ConvertZhlToLlzkFeltPass>();
}

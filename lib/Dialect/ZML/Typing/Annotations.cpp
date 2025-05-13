//===- Annotations.cpp - Type binding attr annotation -----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <zklang/Dialect/ZML/Typing/Annotations.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LogicalResult.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZHL/Typing/Analysis.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>

using namespace zml;
using namespace mlir;
using namespace zhl;

static void annotateOp(
    Operation *op, StringAttr key, const TypeBinding &binding, MLIRContext *ctx,
    const TypeBindings &bindings
) {
  auto attr = FixedTypeBindingAttr::get(ctx, binding, bindings);
  op->setDiscardableAttr(key, attr);
}

static StringAttr defaultKey(Builder &builder) { return builder.getStringAttr("zml.binding"); }

static std::pair<Operation *, StringAttr> findAnnotationTargetForValue(const Value value) {
  Builder builder(value.getContext());

  if (auto result = llvm::dyn_cast<OpResult>(value)) {
    return {result.getDefiningOp(), defaultKey(builder)};
  }

  if (auto arg = llvm::dyn_cast<BlockArgument>(value)) {
    auto block = arg.getOwner();
    assert(block->isEntryBlock() && "only entry blocks supported for annotation");
    return {
        block->getParentOp(), builder.getStringAttr("zml.arg_binding." + Twine(arg.getArgNumber()))
    };
  }

  return {nullptr, nullptr};
}

void zml::annotateOperations(MLIRContext *ctx, ModuleOp mod, const ZIRTypeAnalysis &analysis) {
  auto &bindings = analysis.getBindings();
  Builder builder(ctx);

  for (auto &[value, binding] : analysis.exprs()) {
    if (failed(binding)) {
      continue; // Don't inject on failing checks
    }
    auto [op, key] = findAnnotationTargetForValue(value);
    if (op) {
      assert(key);
      annotateOp(op, key, *binding, ctx, bindings);
    }
  }
  for (auto &[op, binding] : analysis.stmts()) {
    if (succeeded(binding)) {
      annotateOp(op, defaultKey(builder), *binding, ctx, bindings);
    }
  }
  /// ComponentOp instances are not listed in stmts()
  for (auto op : mod.getRegion().getOps<zirgen::Zhl::ComponentOp>()) {
    auto binding = analysis.getType(op.getName());
    if (succeeded(binding)) {
      annotateOp(op, defaultKey(builder), *binding, ctx, bindings);
    }
  }
}

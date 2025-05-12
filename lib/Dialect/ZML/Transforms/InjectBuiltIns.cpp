//===- InjectBuiltIns.cpp - Builtin IR injection ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the implementation of the --inject-builtins pass.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZML/BuiltIns/BuiltIns.h>
#include <zklang/Dialect/ZML/Transforms/PassDetail.h>

using namespace mlir;

namespace zml {

namespace {

class InjectBuiltInsPass : public InjectBuiltInsBase<InjectBuiltInsPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    std::unordered_set<std::string_view> definedNames;
    for (auto op : mod.getOps<zirgen::Zhl::ComponentOp>()) {
      definedNames.insert(op.getName());
    }
    assert(mod->hasTrait<OpTrait::SymbolTable>());
    OpBuilder builder(mod.getRegion());
    // TODO: Replace for a LLZKTypeConverter and add an option to setup the field.
    TypeConverter tc;
    addBuiltins(builder, definedNames, tc);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createInjectBuiltInsPass() {
  return std::make_unique<InjectBuiltInsPass>();
}

} // namespace zml

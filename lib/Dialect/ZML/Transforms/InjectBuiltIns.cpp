// Copyright 2024 Veridise, Inc.

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Dialect/ZML/BuiltIns/BuiltIns.h"
#include "zklang/Dialect/ZML/Transforms/PassDetail.h"
#include <cassert>
#include <llvm/Support/Debug.h>

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
    addBuiltins(builder, definedNames);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createInjectBuiltInsPass() {
  return std::make_unique<InjectBuiltInsPass>();
}

} // namespace zml

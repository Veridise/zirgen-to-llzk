// Copyright 2024 Veridise, Inc.

#include "ZirToZkir/Dialect/ZMIR/BuiltIns/BuiltIns.h"
#include "ZirToZkir/Dialect/ZMIR/Transforms/PassDetail.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include <cassert>
#include <llvm/Support/Debug.h>

using namespace mlir;

namespace zkc::Zmir {

namespace {

class InjectBuiltInsPass : public InjectBuiltInsBase<InjectBuiltInsPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    assert(mod->hasTrait<OpTrait::SymbolTable>());
    OpBuilder builder(mod.getRegion());
    addBuiltins(builder, [&](mlir::StringRef name) {
      return false;

      // The code below doesn't really work and I don't know why
      auto op = mlir::SymbolTable::lookupSymbolIn(mod.getOperation(), name);
      if (op != nullptr) {
        mod->emitError() << "component " << name << " has been overriden";
      } else {
        mod->emitError() << "component " << name << " has not been overriden";
      }
      return op != nullptr;
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createInjectBuiltInsPass() {
  return std::make_unique<InjectBuiltInsPass>();
}

} // namespace zkc::Zmir

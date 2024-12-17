// Copyright 2024 Veridise, Inc.

#include "OpBindings.h"
#include "Scope.h"
#include "Typing.h"
#include "ZirToZkir/Dialect/ZHL/Typing/PassDetail.h"
#include "ZirToZkir/Dialect/ZHL/Typing/TypeBindings.h"
#include "ZirToZkir/Dialect/ZMIR/BuiltIns/BuiltIns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include <cassert>
#include <llvm/Support/Debug.h>
#include <numeric>

using namespace zirgen::Zhl;
using namespace mlir;

namespace zhl {

namespace {

class PrintTypeBindingsPass : public PrintTypeBindingsBase<PrintTypeBindingsPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    TypeBindings bindings;
    zkc::Zmir::addBuiltinBindings(bindings);

    auto result = typeCheck(mod, bindings, zhlTypingRules(bindings));
    result->dump();
    if (mlir::failed(*result)) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createPrintTypeBindingsPass() {
  return std::make_unique<PrintTypeBindingsPass>();
}

} // namespace zhl

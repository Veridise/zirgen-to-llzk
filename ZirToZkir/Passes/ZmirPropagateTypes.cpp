
// Copyright 2024 Veridise, Inc.

#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include "ZirToZkir/Passes/PassDetail.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include <algorithm>
#include <cassert>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;

namespace zkc {

namespace {

struct DeducedType {
  mlir::Type type;

  static DeducedType join(const DeducedType &lhs, const DeducedType &rhs) {
    return lhs; // TODO
  }

  bool operator==(const DeducedType &rhs) const { return type == rhs.type; }

  void print(llvm::raw_ostream &os) const { os << type; }
};

using TypeLattice = dataflow::Lattice<DeducedType>;

class TypePropAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<TypeLattice> {

public:
  using dataflow::SparseForwardDataFlowAnalysis<
      TypeLattice>::SparseForwardDataFlowAnalysis;

  void visitOperation(Operation *op, ArrayRef<const TypeLattice *> operands,
                      ArrayRef<TypeLattice *> results) override {
    llvm::dbgs() << "Visiting " << *op << "...\n";
  }

  /// What am I supposed to do with this function???
  void setToEntryState(TypeLattice *lattice) override {
    llvm::dbgs() << "Set to entry state called\n";
  }
};

class TypePropAnalysisDriver {
public:
  explicit TypePropAnalysisDriver(mlir::Operation *op)
      : cfg(), solver(cfg), analysis(solver), result(mlir::failure()) {
    cfg.setInterprocedural(false);
    result = solver.initializeAndRun(op);
  }

private:
  DataFlowConfig cfg;
  DataFlowSolver solver;
  TypePropAnalysis analysis;
  LogicalResult result;
};

class ZmirPropagateTypesPass
    : public ZmirPropagateTypesBase<ZmirPropagateTypesPass> {

  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<OperationPass<Zmir::ComponentOp>>
createZmirPropagateTypesPass() {
  return std::make_unique<ZmirPropagateTypesPass>();
}

} // namespace zkc

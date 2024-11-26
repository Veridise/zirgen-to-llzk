
// Copyright 2024 Veridise, Inc.

#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include "ZirToZkir/Dialect/ZMIR/Transforms/PassDetail.h"
#include "ZirToZkir/Dialect/ZMIR/Typing/ZMIRTypeConverter.h"
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

namespace zkc::Zmir {

namespace {

struct DeducedType {
  mlir::Type type;

  static DeducedType join(const DeducedType &lhs, const DeducedType &rhs) {
    if (lhs.isPendingType())
      return rhs;
    if (rhs.isPendingType())
      return lhs;
    return lhs; // TODO
  }

  bool operator==(const DeducedType &rhs) const { return type == rhs.type; }

  void print(llvm::raw_ostream &os) const { os << type; }

private:
  bool isPendingType() const { return mlir::isa<PendingType>(type); }
};

using TypeLattice = dataflow::Lattice<DeducedType>;

class TypePropAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<TypeLattice> {

public:
  using dataflow::SparseForwardDataFlowAnalysis<
      TypeLattice>::SparseForwardDataFlowAnalysis;

  void visitOperation(Operation *op, ArrayRef<const TypeLattice *> operands,
                      ArrayRef<TypeLattice *> results) override {}

  /// Sets the initial value for a program point.
  void setToEntryState(TypeLattice *lattice) override {}
};

class TypePropAnalysisDriver {
public:
  explicit TypePropAnalysisDriver(mlir::Operation *op)
      : cfg(), solver(cfg), analysis(solver.load<TypePropAnalysis>()),
        result(mlir::failure()) {
    /*cfg.setInterprocedural(true);*/

    result = solver.initializeAndRun(op);
  }

  LogicalResult getResult() { return result; }

  template <typename Op> void lookup(Op op) {
    auto lattice = solver.lookupState<TypeLattice>(op);
    if (!lattice) {
      return;
    }
  }

  void lookup(mlir::Operation *op) {
    auto lattice = solver.lookupState<TypeLattice>(op);
    if (!lattice) {
      return;
    }
  }

private:
  DataFlowConfig cfg;
  DataFlowSolver solver;
  TypePropAnalysis *analysis;
  LogicalResult result;
};

class FoldUnrealizedCasts
    : public OpConversionPattern<mlir::UnrealizedConversionCastOp> {
public:
  using OpConversionPattern<
      mlir::UnrealizedConversionCastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto val = findValidValue(op.getOperands());
    if (mlir::failed(val))
      return mlir::failure();

    rewriter.replaceAllUsesWith(op.getResults(), *val);
    rewriter.eraseOp(op);

    return mlir::success();
  }

  FailureOr<mlir::ValueRange> findValidValue(ValueRange v) const {
    if (v.size() != 1)
      return mlir::failure();

    auto t = v[0].getType();
    if (isValidZmirType(t) && !mlir::isa<PendingType>(t))
      return v;

    auto op = v[0].getDefiningOp();
    if (op)
      return findValidValue(op->getOperands());
    return mlir::failure();
  }
};

class LegalizeTypesPass : public LegalizeTypesBase<LegalizeTypesPass> {

  void runOnOperation() override {

    auto op = getOperation();

    ZMIRTypeConverter typeConverter;
    mlir::MLIRContext *ctx = op->getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<FoldUnrealizedCasts>(typeConverter, ctx);

    // Set conversion target
    mlir::ConversionTarget target(*ctx);
    target.addLegalOp<mlir::ModuleOp>();
    auto legalIfNoPendingTypes = [](mlir::Operation *op) {
      auto resultTypes = op->getResultTypes();
      auto operandTypes = op->getOperandTypes();
      auto regions = op->getRegions();
      auto attrs = op->getAttrs();

      std::function<bool(Type)> hasPendingTypes = [&](Type t) {
        if (mlir::isa<PendingType>(t))
          return true;

        if (auto funcType = mlir::dyn_cast<FunctionType>(t)) {
          auto res = funcType.getResults();
          auto outputsHavePendingTypes =
              std::any_of(res.begin(), res.end(), hasPendingTypes);
          auto ins = funcType.getInputs();
          auto inputsHavePendingTypes =
              std::any_of(ins.begin(), ins.end(), hasPendingTypes);
          return outputsHavePendingTypes || inputsHavePendingTypes;
        }

        if (auto varsArgs = mlir::dyn_cast<VarArgsType>(t))
          return hasPendingTypes(varsArgs.getInner());

        if (auto array = mlir::dyn_cast<ArrayType>(t))
          return hasPendingTypes(array.getInnerType());

        return false;
      };

      auto resultsHavePendingType =
          std::any_of(resultTypes.begin(), resultTypes.end(), hasPendingTypes);
      auto operandsHavePendingType = std::any_of(
          operandTypes.begin(), operandTypes.end(), hasPendingTypes);
      auto regionsHavePendingType =
          std::any_of(regions.begin(), regions.end(), [&](auto &region) {
            return std::any_of(region.begin(), region.end(), [&](Block &block) {
              auto args = block.getArgumentTypes();
              return std::any_of(args.begin(), args.end(), hasPendingTypes);
            });
          });

      auto attrsHavePendingType =
          std::any_of(attrs.begin(), attrs.end(), [&](NamedAttribute attr) {
            auto attrVal = attr.getValue();
            if (auto typeAttr = mlir::dyn_cast<TypeAttr>(attrVal)) {
              return hasPendingTypes(typeAttr.getValue());
            }
            return false;
          });

      return !(operandsHavePendingType || resultsHavePendingType ||
               regionsHavePendingType || attrsHavePendingType);
    };
    target.addDynamicallyLegalOp<mlir::UnrealizedConversionCastOp>(
        legalIfNoPendingTypes);
    target.addDynamicallyLegalDialect<ZmirDialect, mlir::func::FuncDialect>(
        legalIfNoPendingTypes);

    // Call partialTransformation
    if (mlir::failed(
            mlir::applyFullConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<Zmir::ComponentOp>> createLegalizeTypesPass() {
  return std::make_unique<LegalizeTypesPass>();
}

} // namespace zkc::Zmir

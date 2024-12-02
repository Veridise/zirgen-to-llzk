
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
#include <utility>

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

class UpdateFieldType : public OpConversionPattern<FieldDefOp> {
public:
  template <typename... Args>
  UpdateFieldType(ComponentOp component, SymbolTableCollection &st,
                  Operation *stRoot, Args &&...args)
      : OpConversionPattern<FieldDefOp>(std::forward<Args>(args)...), st(st),
        stRoot(stRoot), component(component) {}

  LogicalResult
  matchAndRewrite(FieldDefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> types;
    collectWriteTypes(op.getName(), types);

    // FIXME: It would be more correct to select the least general type
    // But it is easier with the current implementation to default to either a
    // primitive or the root component.
    if (types.empty())
      return failure();

    auto type = selectType(types.front());
    if (failed(type))
      return op.emitOpError()
             << "could not obtain a type for field '" << op.getName() << "'";

    rewriter.replaceOpWithNewOp<FieldDefOp>(op, adaptor.getSymNameAttr(),
                                            TypeAttr::get(*type));

    return success();
  }

private:
  FailureOr<Type> selectType(Type type) const {
    if (auto compType = mlir::dyn_cast<ComponentType>(type)) {
      if (compType.isRoot())
        return compType;
      auto comp = compType.getDefinition(st, stRoot);
      auto t = comp.getSuperType();
      if (failed(t))
        return t;
      return selectType(*t);
    }

    return type;
  }

  func::FuncOp getBody() const {
    // Interface methods cannot be made const so I have to resort to this
    return const_cast<ComponentOp *>(&component)->getBodyFunc();
  }

  void collectWriteTypes(StringRef fieldName, SmallVector<Type> &types) const {
    getBody().walk([&](WriteFieldOp write) {
      if (write.getFieldName() == fieldName)
        types.push_back(
            findTypeInUseDefChain(write.getVal(), getTypeConverter())
                .getType());
    });
  }

  SymbolTableCollection &st;
  Operation *stRoot;
  ComponentOp component;
};

class LegalizeExecuteRegionOp
    : public OpConversionPattern<scf::ExecuteRegionOp> {
public:
  using OpConversionPattern<scf::ExecuteRegionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ExecuteRegionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getResults().size() != 1)
      return failure();
    SmallVector<Type> types;
    op.walk([&](scf::YieldOp yield) {
      assert(yield.getResults().size() == 1);
      types.push_back(
          findTypeInUseDefChain(yield.getResults().front(), getTypeConverter())
              .getType());
    });

    for (auto t : types) {
      llvm::dbgs() << "Type: " << t << "\n";
    }

    assert(!types.empty());
    rewriter.modifyOpInPlace(op,
                             [&]() { op.getResult(0).setType(types.front()); });
    /*auto newExec = rewriter.replaceOpWithNewOp<scf::ExecuteRegionOp>(op,
     * types);*/
    /*rewriter.inlineRegionBefore(op.getRegion(), newExec.getRegion(),*/
    /*                            newExec.getRegion().end());*/

    /*llvm::dbgs() << newExec << "\n";*/
    return success();
  }
};

class LegalizeWriteArrayOp : public OpConversionPattern<WriteArrayOp> {
public:
  using OpConversionPattern<WriteArrayOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WriteArrayOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arr = findTypeInUseDefChain(adaptor.getArray(), getTypeConverter());
    auto arrType = dyn_cast<ArrayType>(arr.getType());
    assert(arrType);

    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), TypeRange(arrType.getInnerType()),
        ValueRange(adaptor.getValue()));
    rewriter.replaceOpWithNewOp<WriteArrayOp>(op, arr, adaptor.getIndices(),
                                              cast.getResult(0));
    return success();
  }
};

class LegalizeTypesPass : public LegalizeTypesBase<LegalizeTypesPass> {

  void runOnOperation() override {
    SymbolTableCollection st;
    auto op = getOperation();

    ZMIRTypeConverter typeConverter;
    mlir::MLIRContext *ctx = op->getContext();
    mlir::RewritePatternSet patterns(ctx);

    auto mod = op->getParentOfType<ModuleOp>();
    if (!mod) {
      op.emitOpError("could not obtain the module from the component");
      signalPassFailure();
      return;
    }
    patterns.add<UpdateFieldType>(op, st, mod.getOperation(), typeConverter,
                                  ctx);
    patterns.add<FoldUnrealizedCasts, LegalizeExecuteRegionOp,
                 LegalizeWriteArrayOp>(typeConverter, ctx);

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
    target.addDynamicallyLegalDialect<ZmirDialect, mlir::func::FuncDialect,
                                      scf::SCFDialect>(legalIfNoPendingTypes);
    target.addLegalDialect<index::IndexDialect>(); // These will never hold a
                                                   // pending type

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

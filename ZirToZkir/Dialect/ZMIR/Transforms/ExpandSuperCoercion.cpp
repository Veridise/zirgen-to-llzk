

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
#include <memory>
#include <mlir/Pass/Pass.h>

using namespace mlir;

namespace zkc::Zmir {

namespace {

struct CoercionCheckCtx;

class Transform {
public:
  virtual ~Transform() = default;

  virtual Operation *apply(Value, ConversionPatternRewriter &,
                           Location) const = 0;
};

class PrimitiveToRoot : public Transform {
public:
  /// Creates a constructor call to the root component ignoring the input value.
  Operation *apply(Value, ConversionPatternRewriter &rewriter,
                   Location loc) const override {
    auto rootType = ComponentType::getRoot(rewriter.getContext());
    auto root = rootType.getDefinition(st, stRoot);
    assert(root && "root component is missing");
    auto constrRef = rewriter.create<ConstructorRefOp>(loc, root);
    return rewriter.create<func::CallIndirectOp>(loc, constrRef);
  }

  static std::unique_ptr<Transform> create(CoercionCheckCtx &ctx);

  PrimitiveToRoot(SymbolTableCollection &st, Operation *stRoot)
      : st(st), stRoot(stRoot) {}

private:
  SymbolTableCollection &st;
  Operation *stRoot;
};

class ReadSuperField : public Transform {
public:
  /// Using the input value creates a field read op to the super component.
  Operation *apply(Value from, ConversionPatternRewriter &rewriter,
                   Location loc) const override {
    return rewriter.create<ReadFieldOp>(loc, type, from, name);
  }

  static std::unique_ptr<Transform> create(Type type, FlatSymbolRefAttr name) {
    return std::make_unique<ReadSuperField>(type, name);
  }

  ReadSuperField(Type type, FlatSymbolRefAttr name) : type(type), name(name) {}

private:
  Type type;
  FlatSymbolRefAttr name;
};

using TransformList = std::vector<std::unique_ptr<Transform>>;

struct CoercionCheckCtx {
  TransformList transforms;
  SymbolTableCollection st;
  Operation *stRoot;

  CoercionCheckCtx(UnrealizedConversionCastOp op) {
    stRoot = op->getParentOfType<ModuleOp>();
    assert(stRoot && "op doesn't have a module (grand-)parent");
  }
};

std::unique_ptr<Transform> PrimitiveToRoot::create(CoercionCheckCtx &ctx) {
  return std::make_unique<PrimitiveToRoot>(ctx.st, ctx.stRoot);
}

class TransformForcedCasts
    : public OpConversionPattern<UnrealizedConversionCastOp> {
public:
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto from = op.getOperands().front();
    if (!from)
      return failure();
    auto goal = op.getResultTypes().front();
    if (!goal)
      return failure();

    CoercionCheckCtx ctx(op);
    auto check = canCoerceToFinalValue(from.getType(), goal, ctx);
    if (failed(check))
      return failure();

    Value last = from;

    for (auto &t : ctx.transforms) {
      auto *newOp = t->apply(last, rewriter, op.getLoc());
      assert(newOp && "failed generating the transformation");
      last = newOp->getResult(0);
    }

    rewriter.replaceAllUsesWith(op.getResults().front(), last);
    rewriter.eraseOp(op);

    return success();
  }

  inline bool isComponent(Type t) const { return isa<ComponentType>(t); }

  inline bool isComponent(Value v) const { return isComponent(v.getType()); }

  inline bool isRootComponent(Type t) const {
    return isComponent(t) && cast<ComponentType>(t).isRoot();
  }

  inline bool isRootComponent(Value v) const {
    return isRootComponent(v.getType());
  }

  inline bool isVal(Type t) const { return isa<ValType>(t); }
  inline bool isVal(Value v) const { return isVal(v.getType()); }

  LogicalResult canCoerceToFinalValue(Type from, Type goal,
                                      CoercionCheckCtx &ctx) const {
    if (!from || !goal)
      return failure(); // Fail in case of nulls
    if (from == goal)
      return success(); // If its the same type we are done.

    // If 'from' is a component
    if (isComponent(from)) {
      //   If is the root component and the goal is a val or another component
      //   we fail. If is the root component and that's the goal then we are
      //   done
      if (isRootComponent(from)) {
        return success(isRootComponent(goal));
      } else {
        // If is another component
        auto compType = cast<ComponentType>(from);
        auto comp = compType.getDefinition(ctx.st, ctx.stRoot);
        // Try extract the super type and if it fails then fail the whole thing
        auto superType = comp.getSuperType();
        if (failed(superType))
          return failure();

        //   If is another component we add a ReadSuperField and recurse with
        //   using the super component.
        ctx.transforms.push_back(
            ReadSuperField::create(*superType, comp.getSuperFieldName()));
        return canCoerceToFinalValue(*superType, goal, ctx);
      }
    }

    // If it's not a component (a primitive type) and the goal is the root add a
    // PrimitiveToRoot transformation and done
    if (isRootComponent(goal)) {
      ctx.transforms.push_back(PrimitiveToRoot::create(ctx));
      return success();
    }
    // In any other case we fail
    return failure();
  }
};

/// Checks wether the cast is forcefully transforming a component
/// into another or a val.
bool doesntHaveForcedConversion(mlir::UnrealizedConversionCastOp op) {
  // Ignore N-M casts where N != 1 and M != 1
  if (op.getOperands().size() != 1 || op.getResults().size() != 1)
    return true;

  auto inType = op.getOperandTypes().front();
  auto outType = op.getResultTypes().front();

  // We want any component as input but not the root component
  if (!mlir::isa<ComponentType>(inType) ||
      mlir::cast<ComponentType>(inType).isRoot())
    return true;

  // We want any component or a val as output
  return !(mlir::isa<ComponentType>(outType) || mlir::isa<ValType>(outType));
}

class ExpandSuperCoercionPass
    : public ExpandSuperCoercionBase<ExpandSuperCoercionPass> {

  void runOnOperation() override {

    auto op = getOperation();

    ZMIRTypeConverter typeConverter;
    mlir::MLIRContext *ctx = op->getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<TransformForcedCasts>(typeConverter, ctx);

    // Set conversion target
    mlir::ConversionTarget target(*ctx);
    target.addLegalOp<mlir::ModuleOp>();
    target.addDynamicallyLegalOp<mlir::UnrealizedConversionCastOp>(
        doesntHaveForcedConversion);
    target.addLegalDialect<ZmirDialect, mlir::func::FuncDialect>();

    // Call partialTransformation
    if (mlir::failed(
            mlir::applyFullConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<Zmir::ComponentOp>>
createExpandSuperCoercionPass() {
  return std::make_unique<ExpandSuperCoercionPass>();
}

} // namespace zkc::Zmir

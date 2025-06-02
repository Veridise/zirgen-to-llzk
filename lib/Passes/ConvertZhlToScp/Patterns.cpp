//===- Patterns.cpp - ZHL->SCP conversion patterns --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the available conversion patterns for converting ZHL
// component operations into standard SCP dialect operations.
//
//===----------------------------------------------------------------------===//

#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Passes/ConvertZhlToScp/Patterns.h>

#include <llvm/ADT/SmallVector.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Bool/IR/Ops.h>
#include <llzk/Dialect/Cast/IR/Ops.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZHL/Typing/ArrayFrame.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>
#include <zklang/Dialect/ZHL/ZhlOpConversionPattern.h>
#include <zklang/Dialect/ZML/BuiltIns/BuiltIns.h>
#include <zklang/Dialect/ZML/Utils/Helpers.h>

using namespace mlir;
using namespace zirgen::Zhl;
using namespace zhl;
using namespace llzk::array;
using namespace llzk::cast;
using namespace llzk::boolean;

using RangeOpBinding = ZhlOpConversionPattern<RangeOp>::Binding;
using RangeOpBindingsAdaptor = ZhlOpConversionPattern<RangeOp>::BindingsAdaptor;

template <typename... Parents> bool legalSuperOp(SuperOp op) {
  return !mlir::isa<Parents...>(op->getParentOp());
}

static bool legalRangeOp(RangeOp, RangeOpBindingsAdaptor bindings) {
  return bindings.getStart()->isKnownConst() && bindings.getEnd()->isKnownConst();
}

static bool legalRangeOp(RangeOp op) {
  auto bindings = zhl::bindingsAdaptor(op);
  if (failed(bindings)) {
    return true;
  }
  return legalRangeOp(op, *bindings);
}

static bool lessIndicesThanDims(TypedValue<ArrayType> array, ValueRange indices) {
  return indices.size() < array.getType().getDimensionSizes().size();
}

/// Creates the necessary op for reading from an array. If the given number of indices corresponds
/// with the number of dimensions it creates a ReadArrayOp, if not creates a ExtractArrayOp.
static Value readFromArray(
    Location location, Type type, TypedValue<ArrayType> array, ValueRange indices,
    OpBuilder &builder
) {
  if (lessIndicesThanDims(array, indices)) {
    return builder.create<ExtractArrayOp>(location, type, array, indices);
  };

  return builder.create<ReadArrayOp>(location, type, array, indices);
}

/// Creates the necessary op for writing into an array. If the given number of indices corresponds
/// with the number of dimensions it creates a WriteArrayOp, if not creates a InsertArrayOp.
static void writeIntoArray(
    Location location, TypedValue<ArrayType> array, Value value, ValueRange indices,
    OpBuilder &builder
) {
  if (lessIndicesThanDims(array, indices)) {
    builder.create<InsertArrayOp>(location, array, indices, value);
  } else {
    builder.create<WriteArrayOp>(location, array, indices, value);
  }
}

static Value superCoerce(Value v, Type t, OpBuilder &builder) {
  if (v.getType() == t) {
    return v;
  }
  return builder.create<zml::SuperCoerceOp>(v.getLoc(), t, v);
};

template <typename T> static T &slot(const TypeBinding &b) {
  return *mlir::cast_if_present<T>(b.getSlot());
}

template <typename T> static T &slot(const FailureOr<TypeBinding> &b) {
  return *mlir::cast_if_present<T>(b->getSlot());
}

template <typename T> static T &slot(zml::TypeBindingAttr &b) {
  return *mlir::cast_if_present<T>(b->getSlot());
}

namespace {

//===----------------------------------------------------------------------===//
// LowerBlockOpToScf
//===----------------------------------------------------------------------===//

class LowerBlockOpToScf : public ZhlOpConversionPattern<BlockOp> {
public:
  using ZhlOpConversionPattern<BlockOp>::ZhlOpConversionPattern;

  LogicalResult match(BlockOp, Binding, BindingsAdaptor) const final { return success(); }

  void rewrite(
      BlockOp op, Binding binding, OpAdaptor, BindingsAdaptor, ConversionPatternRewriter &rewriter
  ) const final {
    auto exec = rewriter.create<mlir::scf::ExecuteRegionOp>(op.getLoc(), binding.materializeType());
    rewriter.inlineRegionBefore(op.getRegion(), exec.getRegion(), exec.getRegion().end());
    rewriter.replaceOp(op, exec);
  }
};

//===----------------------------------------------------------------------===//
// LowerMapOpToScf
//===----------------------------------------------------------------------===//

class LowerMapOpToScf : public ZhlOpConversionPattern<MapOp> {
public:
  using ZhlOpConversionPattern<MapOp>::ZhlOpConversionPattern;

  LogicalResult matchAndRewrite(
      MapOp op, Binding binding, OpAdaptor adaptor, BindingsAdaptor bindings,
      ConversionPatternRewriter &rewriter
  ) const final {
    auto loopData = loadLoopData(op, binding, adaptor, bindings, rewriter);
    if (failed(loopData)) {
      return failure();
    }

    rewriter.create<mlir::scf::ForOp>(
        op.getLoc(), loopData->from, loopData->to, loopData->stride, loopData->args,
        [this, &loopData, &op,
         &rewriter](mlir::OpBuilder &, mlir::Location loc, mlir::Value iv, mlir::ValueRange) {
      loopData->frame->setInductionVar(iv);
      auto iterData = this->loadLoopIterData(rewriter, loc, *loopData, iv);

      rewriter.inlineBlockBefore(
          &op.getRegion().front(), iterData.prologue, iterData.prologue->end(), iterData.it
      );
    }
    );
    auto self = op->getParentOfType<zml::SelfOp>().getSelfValue();
    auto comp = op->getParentOfType<llzk::component::StructDefOp>();
    auto name = zml::createSlot(loopData->frame, rewriter, comp, op.getLoc(), *getTypeConverter());
    auto slotType = zml::materializeTypeBinding(getContext(), loopData->frame->getBinding());
    auto val = zml::storeAndLoadSlot(
        *loopData->frame, loopData->output, name, slotType, op.getLoc(), rewriter, self
    );
    rewriter.replaceOp(op, val);

    return success();
  }

private:
  struct LoopData {
    Value input;
    Value output;
    Value from;
    Value to;
    Value stride;
    Type type;
    SmallVector<Value, 1> args;
    ArrayFrame *frame;
  };

  struct LoopIterData {
    Value it;
    Block *prologue;
  };

  FailureOr<LoopData> loadLoopData(
      MapOp op, Binding binding, OpAdaptor adaptor, BindingsAdaptor bindings,
      ConversionPatternRewriter &rewriter
  ) const {
    auto *tc = getTypeConverter();
    assert(tc);
    if (!binding->isArray()) {
      return op->emitOpError() << "was expecting 'Array' but got '" << binding->getName() << "'";
    }
    auto *arrayFrame = mlir::dyn_cast_if_present<ArrayFrame>(binding->getSlot());
    if (!arrayFrame) {
      return failure();
    }
    auto inner = bindings.getArray().getArrayElementAttr([&]() { return op.emitError(); });
    if (failed(inner)) {
      return failure();
    }

    auto arrValue = zml::CastHelper::getCastedValue(adaptor.getArray(), rewriter);
    if (failed(arrValue)) {
      return failure();
    }
    auto concreteArrValue =
        coerceToArray(mlir::dyn_cast<TypedValue<zml::ComponentLike>>(*arrValue), rewriter);
    assert(succeeded(concreteArrValue));

    auto arrAlloc = rewriter.create<CreateArrayOp>(
        op.getLoc(), mlir::cast<ArrayType>(tc->convertType(binding.materializeType()))
    );
    auto one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto len = rewriter.create<zml::GetArrayLenOp>(op.getLoc(), *concreteArrValue);

    return LoopData{
        .input = *concreteArrValue,
        .output = arrAlloc,
        .from = zero,
        .to = len.getResult(),
        .stride = one,
        .type = inner->materializeType(),
        .args = {arrAlloc},
        .frame = arrayFrame,
    };
  }

  LoopIterData
  loadLoopIterData(OpBuilder &builder, Location loc, const LoopData &loopData, Value iv) const {
    auto itVal =
        readFromArray(loc, loopData.type, cast<TypedValue<ArrayType>>(loopData.input), iv, builder);
    // Cast it to a zhl Expr type for the block inlining
    auto itValCast =
        builder.create<UnrealizedConversionCastOp>(loc, ExprType::get(getContext()), itVal);

    return {.it = itValCast.getResult(0), .prologue = builder.getInsertionBlock()};
  }
};

//===----------------------------------------------------------------------===//
// LowerRangeOpToScf
//===----------------------------------------------------------------------===//

class LowerRangeOpToScf : public ZhlOpConversionPattern<RangeOp> {
public:
  using ZhlOpConversionPattern<RangeOp>::ZhlOpConversionPattern;

  LogicalResult matchAndRewrite(
      RangeOp op, Binding binding, OpAdaptor adaptor, BindingsAdaptor bindings,
      ConversionPatternRewriter &rewriter
  ) const final {
    auto loopData = getLoopData(op, binding, adaptor, bindings, rewriter);
    if (failed(loopData)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mlir::scf::ForOp>(
        op, loopData->from, loopData->to, loopData->stride, loopData->args,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange args) {
      auto conv = builder.create<IntToFeltOp>(loc, loopData->inner, iv);
      builder.create<WriteArrayOp>(loc, args[0], iv, conv);
      builder.create<scf::YieldOp>(loc, args[0]);
    }
    );

    return mlir::success();
  }

private:
  struct LoopData {
    Value output;
    Value from;
    Value to;
    Value stride;
    SmallVector<Value, 1> args;
    Type inner;
  };

  FailureOr<LoopData> getLoopData(
      RangeOp op, Binding binding, OpAdaptor adaptor, BindingsAdaptor bindings,
      ConversionPatternRewriter &rewriter
  ) const {
    auto *tc = getTypeConverter();
    assert(tc);
    if (!legalRangeOp(op, bindings)) {
      return failure();
    }

    auto inner = binding.getArrayElementAttr([&]() { return op->emitError(); });
    if (failed(inner)) {
      return failure();
    }

    auto Val = zml::builtins::Val(getContext());
    Value startVal =
        zml::CastHelper::getCastedValue(adaptor.getStart(), *bindings.getStart(), rewriter, Val);

    Value endVal =
        zml::CastHelper::getCastedValue(adaptor.getEnd(), *bindings.getEnd(), rewriter, Val);

    auto arrAlloc = rewriter.create<CreateArrayOp>(
        op.getLoc(), mlir::cast<ArrayType>(tc->convertType(binding.materializeType()))
    );

    auto one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    auto start = rewriter.create<FeltToIndexOp>(op.getStart().getLoc(), Val, startVal);
    auto end = rewriter.create<FeltToIndexOp>(op.getEnd().getLoc(), Val, endVal);

    return LoopData{
        .output = arrAlloc,
        .from = start,
        .to = end,
        .stride = one,
        .args = {arrAlloc},
        .inner = inner->materializeType()
    };
  }
};

//===----------------------------------------------------------------------===//
// LowerReduceOpToScf
//===----------------------------------------------------------------------===//

class LowerReduceOpToScf : public ZhlOpConversionPattern<ReduceOp> {
public:
  using ZhlOpConversionPattern<ReduceOp>::ZhlOpConversionPattern;

  LogicalResult matchAndRewrite(
      ReduceOp op, Binding binding, OpAdaptor adaptor, BindingsAdaptor bindings,
      ConversionPatternRewriter &rewriter
  ) const final {
    auto tc = getTypeConverter();
    assert(tc);
    FunctionType constructorType;
    auto inner = bindings.getArray().getArrayElementAttr([&] { return op->emitError(); });
    auto ctorBuilder = initializeCtorCallBuilder(op, binding, bindings, rewriter);

    if (failed(inner) || failed(validateConstructorType(op, bindings.getType(), constructorType)) ||
        failed(ctorBuilder)) {
      return failure();
    }

    auto outputType = binding.materializeType();

    LoopValues lv = prepareLoopValues(op, adaptor, rewriter, outputType, bindings);

    rewriter.replaceOpWithNewOp<scf::ForOp>(
        op, lv.from, lv.to, lv.stride, lv.init,
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {
      slot<ArrayFrame>(binding).setInductionVar(iv);

      auto rhs = superCoerce(
          readFromArray(
              loc, inner->materializeType(), mlir::cast<TypedValue<ArrayType>>(lv.array), iv,
              builder
          ),
          constructorType.getInput(1), builder
      );

      auto lhs = superCoerce(args[0], constructorType.getInput(0), builder);
      auto accResult = ctorBuilder->build(builder, adaptor.getType().getLoc(), {lhs, rhs}, *tc);

      builder.create<scf::YieldOp>(loc, superCoerce(accResult, outputType, builder));
    }
    );

    return success();
  }

private:
  struct LoopValues {
    Value init, array, stride, from, to;
  };

  LogicalResult
  validateConstructorType(Operation *op, Binding accBinding, FunctionType &constructorType) const {
    constructorType = accBinding.materializeCtorType();
    assert(constructorType);
    if (constructorType.getInputs().size() != 2) {
      return op->emitOpError() << "was expecting a constructor with two arguments but got "
                               << constructorType.getInputs().size() << " arguments";
    }
    return success();
  }

  LoopValues prepareLoopValues(
      ReduceOp op, OpAdaptor adaptor, OpBuilder &builder, Type outputType, BindingsAdaptor bindings
  ) const {

    auto initResult = zml::CastHelper::getCastedValue(
        adaptor.getInit(), *bindings.getInit(), builder, outputType
    );

    auto arrayResult = coerceToArray(
        mlir::cast<TypedValue<zml::ComponentLike>>(
            zml::CastHelper::getCastedValue(adaptor.getArray(), *bindings.getArray(), builder)
        ),
        builder
    );
    // If we cannot coerce to an array here either the IR is malformed or we are lacking checks in
    // the type analysis.
    assert(succeeded(arrayResult));

    return {
        .init = initResult,
        .array = *arrayResult,
        .stride = builder.create<arith::ConstantIndexOp>(op.getLoc(), 1),
        .from = builder.create<arith::ConstantIndexOp>(op.getLoc(), 0),
        .to = builder.create<ArrayLengthOp>(op.getLoc(), *arrayResult)
    };
  }

  FailureOr<zml::CtorCallBuilder> initializeCtorCallBuilder(
      Operation *op, Binding binding, BindingsAdaptor &bindings, OpBuilder &builder
  ) const {
    auto self = op->getParentOfType<zml::SelfOp>().getSelfValue();

    return zml::CtorCallBuilder::Make(
        op, *bindings.getType(), builder, self, binding->getContext()
    );
  }
};

//===----------------------------------------------------------------------===//
// LowerSuperOpInBlockOpToScf
//===----------------------------------------------------------------------===//

class LowerSuperOpInBlockOpToScf : public ZhlOpConversionPattern<SuperOp> {
public:
  using ZhlOpConversionPattern<SuperOp>::ZhlOpConversionPattern;

  LogicalResult match(SuperOp op, Binding, BindingsAdaptor) const final {
    return failure(legalSuperOp<scf::ExecuteRegionOp>(op));
  }

  void rewrite(
      SuperOp op, Binding binding, OpAdaptor adaptor, BindingsAdaptor bindings,
      ConversionPatternRewriter &rewriter
  ) const final {
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(
        op, createYieldValue(op, binding, adaptor, bindings, rewriter)
    );
  }

private:
  Value createYieldValue(
      SuperOp op, Binding binding, OpAdaptor adaptor, BindingsAdaptor bindings,
      ConversionPatternRewriter &rewriter
  ) const {
    if (binding->hasClosure()) {
      auto self = op->getParentOfType<zml::SelfOp>().getSelfValue();
      assert(self);
      auto pod = zml::constructPODComponent(op, *binding, rewriter, self, [&]() -> Value {
        return zml::CastHelper::getCastedValue(
            adaptor.getValue(), *bindings.getValue(), rewriter,
            binding.getSuperTypeAttr().materializeType()
        );
      }, binding->getContext(), *getTypeConverter());
      assert(succeeded(pod));
      return *pod;
    }
    return zml::CastHelper::getCastedValue(
        adaptor.getValue(), *bindings.getValue(), rewriter, binding.materializeType()
    );
  }
};

//===----------------------------------------------------------------------===//
// LowerSuperOpInMapOpToScf
//===----------------------------------------------------------------------===//

class LowerSuperOpInMapOpToScf : public ZhlOpConversionPattern<SuperOp> {
public:
  using ZhlOpConversionPattern<SuperOp>::ZhlOpConversionPattern;

  LogicalResult match(SuperOp op, Binding, BindingsAdaptor) const final {
    return failure(legalSuperOp<scf::ForOp>(op));
  }

  void rewrite(
      SuperOp op, Binding, OpAdaptor adaptor, BindingsAdaptor, ConversionPatternRewriter &rewriter
  ) const final {

    auto loopOp = mlir::cast<scf::ForOp>(op->getParentOp());
    auto value = zml::CastHelper::getCastedValue(adaptor.getValue(), rewriter);
    assert(succeeded(value));
    auto iv = loopOp.getInductionVar();
    auto arr = loopOp.getRegionIterArgs().front();

    writeIntoArray(op.getLoc(), mlir::cast<TypedValue<ArrayType>>(arr), *value, iv, rewriter);
    rewriter.create<WriteArrayOp>(op.getLoc(), arr, iv, *value);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, loopOp.getRegionIterArgs());
  }
};

//===----------------------------------------------------------------------===//
// LowerSuperOpInSwitchOpToScf
//===----------------------------------------------------------------------===//

class LowerSuperOpInSwitchOpToScf : public ZhlOpConversionPattern<SuperOp> {
public:
  using ZhlOpConversionPattern<SuperOp>::ZhlOpConversionPattern;

  LogicalResult match(SuperOp op, Binding, BindingsAdaptor) const final {
    return failure(legalSuperOp<scf::IfOp>(op));
  }

  void rewrite(
      SuperOp op, Binding binding, OpAdaptor adaptor, BindingsAdaptor bindings,
      ConversionPatternRewriter &rewriter
  ) const final {
    auto parent = op->getParentOp();

    auto type = binding.materializeType();
    assert(parent->getResultTypes().size() == 1);
    auto parentType = parent->getResultTypes().front();
    auto value =
        zml::CastHelper::getCastedValue(adaptor.getValue(), *bindings.getValue(), rewriter, type);
    auto coercion = rewriter.create<zml::SuperCoerceOp>(op.getLoc(), parentType, value);

    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, coercion.getResult());
  }
};

//===----------------------------------------------------------------------===//
// LowerSwitchOpToScf
//===----------------------------------------------------------------------===//

class LowerSwitchOpToScf : public ZhlOpConversionPattern<SwitchOp> {
public:
  using ZhlOpConversionPattern<SwitchOp>::ZhlOpConversionPattern;

  LogicalResult match(SwitchOp, Binding, BindingsAdaptor) const final { return success(); }

  void rewrite(
      SwitchOp op, Binding binding, OpAdaptor adaptor, BindingsAdaptor bindings,
      ConversionPatternRewriter &rewriter
  ) const final {
    auto tc = getTypeConverter();
    assert(tc);
    auto arrType = zml::builtins::Array(
        zml::builtins::Val(getContext()), rewriter.getIndexAttr(op.getNumRegions()), *tc
    );
    auto selector = zml::CastHelper::getCastedValue(
        adaptor.getSelector(), *bindings.getSelector(), rewriter, arrType
    );
    SmallVector<Value> conds = llvm::map_to_vector(llvm::enumerate(op.getRegions()), [&](auto t) {
      return createNthCond(t.index(), selector, rewriter);
    });

    auto retType = binding.materializeType();

    auto execRegion = rewriter.create<scf::ExecuteRegionOp>(op.getLoc(), retType);
    RegionRange regions = op.getRegions();
    Block &block = execRegion.getRegion().emplaceBlock();
    buildIfThenElseChain(
        regions.begin(), regions.end(), conds.begin(), block, block.end(), rewriter, retType
    );
    rewriter.replaceOp(op, execRegion);
  }

private:
  Value createNthCond(unsigned long idx, Value selector, OpBuilder &rewriter) const {
    auto val = zml::builtins::Val(getContext());
    // Load the selector value from the array
    auto nth = rewriter.create<arith::ConstantIndexOp>(selector.getLoc(), idx);
    auto item = rewriter.create<ReadArrayOp>(selector.getLoc(), val, selector, ValueRange(nth));

    // Check if the value is equal to 1 (by converting it into a boolean)
    return rewriter.create<zml::ValToI1Op>(selector.getLoc(), rewriter.getI1Type(), item);
  }

  /// Inlines a switch arm region. The region must have only 1 block.
  void inlineRegion(
      Region *region, Block &dest, Block::iterator it, ConversionPatternRewriter &rewriter
  ) const {
    assert(region->getBlocks().size() == 1);
    rewriter.inlineBlockBefore(&region->front(), &dest, it);
  }

  /// Builds an if-then-else chain with each region of the switch op.
  template <typename RegionIt, typename ValueIt, typename BlockIt>
  void buildIfThenElseChain(
      RegionIt region_begin, RegionIt region_end, ValueIt conds, Block &dest, BlockIt destIt,
      ConversionPatternRewriter &rewriter, Type retType
  ) const {
    Value cond = *conds;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(&dest, destIt);
    if (std::next(region_begin) == region_end) {
      rewriter.create<AssertOp>(
          cond.getLoc(), cond, rewriter.getStringAttr("no mux branch selected")
      );
      inlineRegion(*region_begin, dest, rewriter.getInsertionPoint(), rewriter);
      return;
    }

    auto ifOp = rewriter.create<scf::IfOp>(cond.getLoc(), retType, cond, true, true);
    inlineRegion(
        *region_begin, ifOp.getThenRegion().front(), ifOp.getThenRegion().front().end(), rewriter
    );
    buildIfThenElseChain(
        std::next(region_begin), region_end, std::next(conds), ifOp.getElseRegion().front(),
        ifOp.getElseRegion().front().end(), rewriter, retType
    );
    rewriter.create<scf::YieldOp>(ifOp.getLoc(), ifOp.getResults());
  }
};

} // namespace

void zklang::populateZhlToScpConversionPatterns(
    const TypeConverter &tc, RewritePatternSet &patterns
) {
  auto *ctx = patterns.getContext();
  patterns.add<
      // clang-format off
      LowerBlockOpToScf, 
      LowerMapOpToScf, 
      LowerRangeOpToScf,
      LowerReduceOpToScf,  
      LowerSuperOpInBlockOpToScf, 
      LowerSuperOpInMapOpToScf, 
      LowerSuperOpInSwitchOpToScf, 
      LowerSwitchOpToScf
      // clang-format on
      >(tc, ctx);
}

void zklang::populateZhlToScpConversionTarget(ConversionTarget &target) {
  target.addLegalDialect<scf::SCFDialect, BuiltinDialect>();
  target.addIllegalOp<MapOp, BlockOp, ReduceOp, SwitchOp>();
  target.addDynamicallyLegalOp<SuperOp>(legalSuperOp<MapOp, BlockOp, SwitchOp>);
  target.addDynamicallyLegalOp<RangeOp>([](RangeOp op) { return legalRangeOp(op); });
}

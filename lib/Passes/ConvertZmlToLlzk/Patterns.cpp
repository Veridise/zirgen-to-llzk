//===- Patterns.cpp - ZML->LLZK conversion patterns -------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Bool/IR/Ops.h>
#include <llzk/Dialect/Cast/IR/Ops.h>
#include <llzk/Dialect/Constrain/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Attrs.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Types.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Global/IR/Ops.h>
#include <llzk/Dialect/LLZK/IR/AttributeHelper.h>
#include <llzk/Dialect/Polymorphic/IR/Ops.h>
#include <llzk/Dialect/String/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Types.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <unordered_set>
#include <vector>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/Types.h>
#include <zklang/Passes/ConvertZmlToLlzk/Patterns.h>

using namespace zml;
using namespace mlir;
using namespace llzk;

LogicalResult LitValOpLowering::matchAndRewrite(
    LitValOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<felt::FeltConstantOp>(
      op, felt::FeltConstAttr::get(getContext(), llvm::APInt(64, adaptor.getValue()))
  );
  return success();
}

LogicalResult LitStrOpLowering::matchAndRewrite(
    LitStrOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<string::LitStringOp>(
      op, string::StringType::get(rewriter.getContext()), adaptor.getValue()
  );
  return success();
}

LogicalResult ComponentLowering::matchAndRewrite(
    SplitComponentOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  auto newOp = rewriter.create<component::StructDefOp>(
      op.getLoc(), op.getNameAttr(),
      op.getParams().has_value() ? *op.getParams() : rewriter.getArrayAttr({})
  );
  rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(), newOp.getRegion().end());
  rewriter.eraseOp(op);
  return success();
}

LogicalResult FieldDefOpLowering::matchAndRewrite(
    FieldDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<component::FieldDefOp>(
      op, op.getNameAttr(), getTypeConverter()->convertType(op.getType()), op.getColumn()
  );
  return success();
}

/// Inspired by FuncOp::cloneInto
void cloneAttrsIntoLlzkFunc(func::FuncOp src, function::FuncDefOp dest) {
  // Add the attributes of this function to dest (except visibility unless is
  // extern).
  llvm::MapVector<StringAttr, Attribute> newAttrMap;
  for (const auto &attr : dest->getAttrs()) {
    newAttrMap.insert({attr.getName(), attr.getValue()});
  }
  for (const auto &attr : src->getAttrs()) {
    newAttrMap.insert({attr.getName(), attr.getValue()});
  }

  if (!newAttrMap.contains(StringAttr::get(src.getContext(), "extern"))) {
    newAttrMap.erase(StringAttr::get(src.getContext(), "sym_visibility"));
  }

  auto newAttrs =
      llvm::to_vector(llvm::map_range(newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
    return NamedAttribute(attrPair.first, attrPair.second);
  }));
  dest->setAttrs(DictionaryAttr::get(src.getContext(), newAttrs));
}

LogicalResult FuncOpLowering::matchAndRewrite(
    func::FuncOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {

  auto type = op.getFunctionType();
  TypeConverter::SignatureConversion result(type.getNumInputs());

  llvm::SmallVector<Type, 1> newResults;
  if (failed(getTypeConverter()->convertSignatureArgs(type.getInputs(), result)) ||
      failed(getTypeConverter()->convertTypes(type.getResults(), newResults)) ||
      failed(rewriter.convertRegionTypes(&op.getFunctionBody(), *getTypeConverter(), &result))) {
    return failure();
  }

  auto newType = FunctionType::get(rewriter.getContext(), result.getConvertedTypes(), newResults);

  auto newFuncOp = rewriter.create<function::FuncDefOp>(op.getLoc(), op.getNameAttr(), newType);
  cloneAttrsIntoLlzkFunc(op, newFuncOp);
  rewriter.inlineRegionBefore(op.getRegion(), newFuncOp.getRegion(), newFuncOp.end());
  rewriter.replaceOp(op, newFuncOp);
  return success();
}

LogicalResult ReturnOpLowering::matchAndRewrite(
    func::ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<function::ReturnOp>(op, adaptor.getOperands());
  return success();
}

LogicalResult ExternCallOpLowering::matchAndRewrite(
    func::CallIndirectOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto calleeOp = adaptor.getCallee().getDefiningOp();
  if (!calleeOp) {
    return failure();
  }
  auto callee = dyn_cast<ExternFnRefOp>(calleeOp);
  if (!callee) {
    return failure();
  }

  llvm::SmallVector<Type> results;

  auto convRes = getTypeConverter()->convertTypes(op.getCallee().getType().getResults(), results);
  if (failed(convRes)) {
    return op->emitError("failed to transform zml types into llzk types");
  }
  rewriter.replaceOpWithNewOp<function::CallOp>(
      op, results, callee.getNameAttr(), adaptor.getCalleeOperands()
  );
  return success();
}

LogicalResult LowerNopOp::matchAndRewrite(
    NopOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  if (adaptor.getIns().size() == op.getNumResults()) {
    rewriter.replaceOp(op, adaptor.getIns());
  } else {
    rewriter.eraseOp(op);
  }
  return success();
}

LogicalResult LowerUnifiableCastOp::matchAndRewrite(
    UnifiableCastOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<polymorphic::UnifiableCastOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getInput()
  );
  return success();
}

// Set of the builtins (by name) that are converted to a LLZK felt or a extended field element
// representation. Used for shortcircuiting the lowering of SuperCoerceOp and for removing the calls
// to @constrain that point to structs that get removed during lowering because the implementation
// of these types gets removed.
static std::unordered_set<std::string_view> builtinsConvertibleToOps{
    "Val", "Add",     "Sub",    "Mul",    "BitAnd", "Inv",    "Isz",    "Neg",
    "Mod", "InRange", "ExtVal", "ExtAdd", "ExtSub", "ExtInv", "ExtMul", "MakeExt"
};

static bool wasConvertedToPrimitiveType(ComponentType t) {
  return t.getBuiltin() &&
         (builtinsConvertibleToOps.find(t.getName().getValue()) != builtinsConvertibleToOps.end());
}

static Value readArray(Value src, Value iv, OpBuilder &builder, Location loc) {
  auto type = llvm::cast<array::ArrayType>(src.getType());
  if (type.getDimensionSizes().size() == 1) {
    return builder.create<array::ReadArrayOp>(loc, src, iv);
  }
  return builder.create<array::ExtractArrayOp>(loc, src, iv);
}

static void writeArray(Value dst, Value iv, Value val, OpBuilder &builder, Location loc) {
  auto type = llvm::cast<array::ArrayType>(dst.getType());
  if (type.getDimensionSizes().size() == 1) {
    builder.create<array::WriteArrayOp>(loc, dst, iv, val);
  } else {
    builder.create<array::InsertArrayOp>(loc, dst, iv, val);
  }
}

static Value readSuperFields(
    ComponentType type, Value chain, Type target, const TypeConverter &tc, Location loc,
    OpBuilder &builder
);

static Value copyArraySuperFields(
    ComponentType innerType, Value chain, Type target, const TypeConverter &tc, Location loc,
    OpBuilder &builder
) {
  auto targetInner = llvm::cast<ComponentType>(target).getArrayInnerType();
  auto targetArrayType = llvm::cast<array::ArrayType>(tc.convertType(target));

  auto array = builder.create<array::CreateArrayOp>(loc, targetArrayType);

  auto lb = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto ub = builder.create<array::ArrayLengthOp>(
      loc, TypeRange({IndexType::get(builder.getContext())}), chain, lb
  );
  auto stride = builder.create<arith::ConstantIndexOp>(loc, 1);

  auto loop = builder.create<scf::ForOp>(
      loc, lb, stride, ub, ValueRange({array}),
      [&](OpBuilder &loopBuilder, Location loopLoc, Value iv, ValueRange args) {
    auto src = readArray(chain, iv, loopBuilder, loopLoc);
    auto super = readSuperFields(innerType, src, *targetInner, tc, loopLoc, loopBuilder);
    writeArray(args[0], iv, super, loopBuilder, loopLoc);

    loopBuilder.create<scf::YieldOp>(loopLoc, args);
  }
  );
  return loop.getResult(0);
}

static Value handleArraySpecialCases(
    ComponentType type, Value chain, Type target, const TypeConverter &tc, Location loc,
    OpBuilder &builder
) {
  auto targetAsComp = llvm::dyn_cast_if_present<ComponentType>(target);
  if (!targetAsComp || !targetAsComp.isConcreteArray()) {
    return chain;
  }

  auto targetLlzkType = llvm::cast<array::ArrayType>(tc.convertType(target));
  // We don't need to create the copying code if the llzk versions of the types are going to unify.
  if (typesUnify(chain.getType(), targetLlzkType)) {
    // Cast the output type to the equivalent target type to avoid an unrealizable conversion cast.
    return builder.create<polymorphic::UnifiableCastOp>(loc, targetLlzkType, chain);
  }

  // If the target is an array component then read the super fields of the inner elements and
  // returns a new array with those values.
  return copyArraySuperFields(
      llvm::cast<ComponentType>(*type.getArrayInnerType()), chain, target, tc, loc, builder
  );
}

static Value readSuperFields(
    ComponentType type, Value chain, Type target, const TypeConverter &tc, Location loc,
    OpBuilder &builder
) {

  auto superComp = llvm::dyn_cast_if_present<ComponentType>(type.getSuperType());
  // Stop if we reached a builtin that transforms to a primitive llzk type, we are done, or we
  // cannot continue extracting
  if (type == target || wasConvertedToPrimitiveType(type) || !superComp ||
      llvm::isa<TypeVarType>(type.getSuperType())) {
    return chain;
  }
  if (type.isConcreteArray()) {
    return handleArraySpecialCases(type, chain, target, tc, loc, builder);
  }

  auto read = builder.create<component::FieldReadOp>(
      loc, tc.convertType(superComp), chain, builder.getStringAttr("$super")
  );
  return readSuperFields(superComp, read, target, tc, loc, builder);
}

LogicalResult LowerSuperCoerceOp::matchAndRewrite(
    SuperCoerceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto t = llvm::dyn_cast<ComponentType>(op.getOperand().getType());
  assert(t);
  auto tc = getTypeConverter();
  assert(tc);
  rewriter.replaceOp(
      op, readSuperFields(
              t, adaptor.getComponent(), op.getResult().getType(), *tc, op.getLoc(), rewriter
          )
  );
  return success();
}

LogicalResult LowerConstrainCallOp::matchAndRewrite(
    ConstrainCallOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto compType = op.getSelf().getType();
  auto comp = llvm::dyn_cast<ComponentType>(compType);
  if (!comp) {
    return op->emitOpError() << "was expecting a component type but got " << compType;
  }
  auto compName = comp.getName().getAttr();
  if (wasConvertedToPrimitiveType(comp)) {
    rewriter.eraseOp(op);
    return success();
  }
  auto sym =
      SymbolRefAttr::get(compName, {SymbolRefAttr::get(rewriter.getStringAttr("constrain"))});
  rewriter.replaceOpWithNewOp<function::CallOp>(op, TypeRange(), sym, adaptor.getOperands());

  return success();
}

LogicalResult LowerLoadValParamOp::matchAndRewrite(
    LoadValParamOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<polymorphic::ConstReadOp>(
      op, felt::FeltType::get(getContext()), SymbolRefAttr::get(op.getParamAttr())
  );
  return success();
}

/// Given an attribute materializes it into a Value if it's either a SymbolRefAttr or an
/// IntegerAttr. Any other kind of Attribute is considered malformed IR and will abort.
static Value materializeParam(Attribute attr, OpBuilder &builder, Location loc) {
  if (auto symAttr = llvm::dyn_cast<SymbolRefAttr>(attr)) {
    return builder.create<polymorphic::ConstReadOp>(
        loc, builder.getIndexType(), symAttr.getRootReference()
    );
  }
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
    return builder.create<arith::ConstantIndexOp>(loc, fromAPInt(intAttr.getValue()));
  }
  assert(false && "Cannot materialize something that is not a symbol or a literal integer");
}

namespace {

struct Params {
  ArrayRef<Attribute> callee, caller;
};

struct AffineParams {
  SmallVector<ConstExprAttr> decl, lifted;

  size_t size() const { return decl.size() + lifted.size(); }
};

} // namespace

static void collectAffineParams(ArrayRef<Attribute> attrs, SmallVectorImpl<ConstExprAttr> &out) {
  for (auto attr : attrs) {
    if (auto param = llvm::dyn_cast<ConstExprAttr>(attr)) {
      out.push_back(param);
    }
  }
}

static ValueRange materializeValuesForParam(
    ConstExprAttr param, ArrayRef<Attribute> sourceParams, SmallVectorImpl<Value> &out,
    OpBuilder &builder, Location loc
) {
  out = llvm::map_to_vector(param.getFormals(), [&](auto formal) {
    return materializeParam(sourceParams[formal], builder, loc);
  });
  return out;
}

static void materializeValuesForParams(
    ArrayRef<ConstExprAttr> affineParams, ArrayRef<Attribute> sourceParams,
    MutableArrayRef<SmallVector<Value>> mem, OpBuilder &builder, Location loc,
    SmallVectorImpl<ValueRange> &out
) {
  auto res = llvm::map_to_vector(llvm::zip_equal(mem, affineParams), [&](auto in) -> ValueRange {
    auto [values, param] = in;
    return materializeValuesForParam(param, sourceParams, values, builder, loc);
  });
  out.insert(out.end(), res.begin(), res.end());
}

LogicalResult CallIndirectOpLoweringInCompute::matchAndRewrite(
    func::CallIndirectOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto parent = op->getParentOfType<function::FuncDefOp>();
  if (!parent || parent.getName() != "compute") {
    return failure(); // Don't operate on non compute calls
  }

  auto callee = llvm::dyn_cast<ConstructorRefOp>(adaptor.getCallee().getDefiningOp());
  if (!callee) {
    return failure();
  }

  Params params{
      .callee = llvm::cast<ComponentType>(op.getResult(0).getType()).getParams(),
      .caller = op->getParentOfType<component::StructDefOp>()
                    .getConstParams()
                    .value_or(rewriter.getArrayAttr({}))
                    .getValue()
  };

  auto numDeclParams = params.callee.size() - callee.getNumLiftedParams().getZExtValue();

  AffineParams affineParams;
  collectAffineParams(params.callee.take_front(numDeclParams), affineParams.decl);
  collectAffineParams(params.callee.drop_front(numDeclParams), affineParams.lifted);

  // Allocate here the values we may generate
  SmallVector<SmallVector<Value>> mapOperandsMem(affineParams.size());
  // This idiom does not use any dimensions
  SmallVector<int32_t> dimsPerMap(affineParams.size(), 0);
  // And store a ValueRange pointing to the vector here
  SmallVector<ValueRange> mapOperands;
  mapOperands.reserve(affineParams.size());

  materializeValuesForParams(
      affineParams.decl, params.caller,
      MutableArrayRef(mapOperandsMem).take_front(affineParams.decl.size()), rewriter, op.getLoc(),
      mapOperands
  );
  materializeValuesForParams(
      affineParams.lifted, params.callee,
      MutableArrayRef(mapOperandsMem).drop_front(affineParams.decl.size()), rewriter, op.getLoc(),
      mapOperands
  );

  auto sym = SymbolRefAttr::get(
      callee.getComponentAttr().getAttr(), {SymbolRefAttr::get(parent.getNameAttr())}
  );

  llvm::SmallVector<Type> types;
  auto convRes = getTypeConverter()->convertTypes(op.getResultTypes(), types);
  if (failed(convRes)) {
    return op->emitError("failed to transform zml types into llzk types");
  }
  ValueRange args(iterator_range(adaptor.getOperands().begin() + 1, adaptor.getOperands().end()));

  rewriter.replaceOpWithNewOp<function::CallOp>(op, types, sym, mapOperands, dimsPerMap, args);
  return success();
}

LogicalResult WriteFieldOpLowering::matchAndRewrite(
    WriteFieldOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<component::FieldWriteOp>(
      op, adaptor.getComponent(), adaptor.getFieldNameAttr(), adaptor.getVal()
  );
  return success();
}

LogicalResult RemoveConstructorRefOp::matchAndRewrite(
    ConstructorRefOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.eraseOp(op);
  return success();
}

LogicalResult RemoveExternFnRefOp::matchAndRewrite(
    ExternFnRefOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.eraseOp(op);
  return success();
}

LogicalResult LowerReadFieldOp::matchAndRewrite(
    ReadFieldOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<component::FieldReadOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getComponent(),
      adaptor.getFieldNameAttr().getAttr()
  );
  return success();
}

LogicalResult LowerConstrainOp::matchAndRewrite(
    ConstrainOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<constrain::EmitEqualityOp>(op, adaptor.getLhs(), adaptor.getRhs());
  return success();
}

LogicalResult LowerInRangeOp::matchAndRewrite(
    InRangeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto le = rewriter.create<boolean::CmpOp>(
      op.getLoc(), boolean::FeltCmpPredicateAttr::get(getContext(), boolean::FeltCmpPredicate::LE),
      adaptor.getLow(), adaptor.getMid()
  );
  auto lt = rewriter.create<boolean::CmpOp>(
      op.getLoc(), boolean::FeltCmpPredicateAttr::get(getContext(), boolean::FeltCmpPredicate::LT),
      adaptor.getMid(), adaptor.getHigh()
  );
  auto felt = felt::FeltType::get(getContext());
  auto convLe = rewriter.create<cast::IntToFeltOp>(op.getLoc(), felt, le);
  auto convLt = rewriter.create<cast::IntToFeltOp>(op.getLoc(), felt, lt);
  auto mul = rewriter.create<felt::MulFeltOp>(op.getLoc(), convLe, convLt);
  rewriter.replaceOp(op, mul);

  return success();
}

LogicalResult LowerNewArrayOp::matchAndRewrite(
    NewArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  // NewArrayOp is SameTypeOperands so by querying the type of any one of the elements we can know
  // if the inputs are scalar or arrays

  auto type = getTypeConverter()->convertType(op.getType());
  auto arrType = dyn_cast<array::ArrayType>(type);
  if (!arrType) {
    return op.emitOpError() << "was expecting an array type";
  }

  // If it's an array then we allocate an empty one and then insert each operand with InsertArrayOp
  if (!adaptor.getElements().empty() && isa<array::ArrayType>(adaptor.getElements()[0].getType())) {
    Location cachedLoc = op.getLoc();
    auto arr = rewriter.replaceOpWithNewOp<array::CreateArrayOp>(op, arrType, ValueRange());
    for (size_t i = 0; i < adaptor.getElements().size(); i++) {
      auto idx = rewriter.create<arith::ConstantIndexOp>(cachedLoc, i);
      rewriter.create<array::InsertArrayOp>(
          cachedLoc, arr, ValueRange({idx}), adaptor.getElements()[i]
      );
    }
  } else {
    rewriter.replaceOpWithNewOp<array::CreateArrayOp>(op, arrType, adaptor.getElements());
  }
  return success();
}

LogicalResult LowerLitValArrayOp::matchAndRewrite(
    LitValArrayOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  auto type = getTypeConverter()->convertType(op.getType());
  auto arrType = dyn_cast<array::ArrayType>(type);
  if (!arrType) {
    return op.emitOpError() << "was expecting an array type";
  }

  felt::FeltType felt = felt::FeltType::get(getContext());
  SmallVector<Value> lits = llvm::map_to_vector(op.getElements(), [&](int64_t value) -> Value {
    return rewriter.create<felt::FeltConstantOp>(
        op.getLoc(), felt, felt::FeltConstAttr::get(getContext(), toAPInt(value))
    );
  });

  rewriter.replaceOpWithNewOp<array::CreateArrayOp>(op, arrType, ValueRange(lits));
  return success();
}

LogicalResult LowerReadArrayOp::matchAndRewrite(
    ReadArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  llvm::SmallVector<Value> toIndexOps =
      llvm::map_to_vector(adaptor.getIndices(), [&](Value index) -> Value {
    if (llvm::isa<IndexType>(index.getType())) {
      return llvm::cast<TypedValue<IndexType>>(index);
    }
    return rewriter.create<cast::FeltToIndexOp>(op.getLoc(), rewriter.getIndexType(), index)
        .getResult();
  });
  auto convertedType = getTypeConverter()->convertType(op.getType());
  if (isa<array::ArrayType>(convertedType)) {
    rewriter.replaceOpWithNewOp<array::ExtractArrayOp>(
        op, convertedType, adaptor.getLvalue(), toIndexOps
    );
  } else {
    rewriter.replaceOpWithNewOp<array::ReadArrayOp>(
        op, convertedType, adaptor.getLvalue(), toIndexOps
    );
  }
  return success();
}

LogicalResult LowerIsz::matchAndRewrite(
    IsZeroOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto zero = rewriter.create<felt::FeltConstantOp>(
      op.getLoc(), felt::FeltConstAttr::get(getContext(), APInt::getZero(64))
  );
  auto cmpOp = rewriter.create<boolean::CmpOp>(
      op.getLoc(), boolean::FeltCmpPredicateAttr::get(getContext(), boolean::FeltCmpPredicate::EQ),
      adaptor.getIn(), zero
  );
  auto felt = felt::FeltType::get(getContext());
  rewriter.replaceOpWithNewOp<cast::IntToFeltOp>(op, felt, cmpOp);
  return success();
}

LogicalResult LowerAllocArrayOp::matchAndRewrite(
    AllocArrayOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  auto type = getTypeConverter()->convertType(op.getType());
  auto arrType = dyn_cast<array::ArrayType>(type);
  if (!arrType) {
    return op.emitOpError() << "was expecting an array type";
  }

  ArrayAttr compParams = op->getParentOfType<component::StructDefOp>().getType().getParams();
  SmallVector<ConstExprAttr> affineArrayParams;
  ComponentType cType = op.getResult().getType();
  for (Attribute attr : cType.getParams()) {
    if (auto param = llvm::dyn_cast<ConstExprAttr>(attr)) {
      affineArrayParams.push_back(param);
    }
  }
  // Allocate here the values we may generate
  SmallVector<SmallVector<Value>> mapOperandsMem(affineArrayParams.size());
  // And store a ValueRange pointing to the vector here
  SmallVector<ValueRange> mapOperands;
  // This idiom does not use any dimensions
  SmallVector<int32_t> dimsPerMap(affineArrayParams.size(), 0);
  mapOperands.reserve(affineArrayParams.size());

  for (auto [idx, constExpr] : llvm::enumerate(affineArrayParams)) {
    auto &values = mapOperandsMem[idx];
    for (uint64_t formal : constExpr.getFormals()) {
      assert(formal <= std::numeric_limits<unsigned int>::max());
      if (static_cast<unsigned int>(formal) >= compParams.size()) {
        return op->emitError() << "requested parameter #" << formal
                               << ", but component in scope only has " << compParams.size()
                               << " parameters";
      }
      values.push_back(
          materializeParam(compParams[static_cast<unsigned int>(formal)], rewriter, op->getLoc())
      );
    }
    mapOperands.push_back(values);
  }

  rewriter.replaceOpWithNewOp<array::CreateArrayOp>(op, arrType, mapOperands, dimsPerMap);

  return success();
}

LogicalResult LowerArrayLengthOp::matchAndRewrite(
    GetArrayLenOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto c = rewriter.create<arith::ConstantOp>(
      op.getLoc(), IndexType::get(getContext()), rewriter.getIndexAttr(0)
  );
  rewriter.replaceOpWithNewOp<array::ArrayLengthOp>(op, adaptor.getArray(), c);

  return success();
}

LogicalResult LowerIndexToValOp::matchAndRewrite(
    IndexToValOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<cast::IntToFeltOp>(
      op, felt::FeltType::get(getContext()), adaptor.getIndex()
  );

  return success();
}

LogicalResult LowerValToIndexOp::matchAndRewrite(
    ValToIndexOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  Value v = adaptor.getVal();
  if (llvm::isa<IndexType>(v.getType())) {
    rewriter.replaceAllUsesWith(op, v);
    rewriter.eraseOp(op);
  } else if (auto constRead =
                 llvm::dyn_cast_if_present<polymorphic::ConstReadOp>(v.getDefiningOp())) {
    // If the Value comes from a ConstReadOp, just directly read as an 'index' rather than cast.
    rewriter.replaceOpWithNewOp<polymorphic::ConstReadOp>(
        op, rewriter.getIndexType(), constRead.getConstNameAttr()
    );
  } else {
    rewriter.replaceOpWithNewOp<cast::FeltToIndexOp>(op, rewriter.getIndexType(), v);
  }
  return success();
}

LogicalResult LowerWriteArrayOp::matchAndRewrite(
    WriteArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  if (llvm::isa<array::ArrayType>(adaptor.getValue().getType())) {
    rewriter.replaceOpWithNewOp<array::InsertArrayOp>(
        op, adaptor.getArray(), adaptor.getIndices(), adaptor.getValue()
    );
  } else {
    rewriter.replaceOpWithNewOp<array::WriteArrayOp>(
        op, adaptor.getArray(), adaptor.getIndices(), adaptor.getValue()
    );
  }

  return success();
}

LogicalResult UpdateScfExecuteRegionOpTypes::matchAndRewrite(
    scf::ExecuteRegionOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  SmallVector<Type> newTypes;
  if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newTypes))) {
    return failure();
  }
  auto exec = rewriter.create<scf::ExecuteRegionOp>(op.getLoc(), newTypes);
  rewriter.inlineRegionBefore(op.getRegion(), exec.getRegion(), exec.getRegion().end());
  rewriter.replaceOp(op, exec);

  return success();
}

LogicalResult ValToI1OpLowering::matchAndRewrite(
    ValToI1Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto zero = rewriter.create<felt::FeltConstantOp>(
      op.getLoc(), felt::FeltType::get(getContext()),
      felt::FeltConstAttr::get(getContext(), APInt(1, 0))
  );
  rewriter.replaceOpWithNewOp<boolean::CmpOp>(
      op, boolean::FeltCmpPredicateAttr::get(getContext(), boolean::FeltCmpPredicate::NE),
      adaptor.getVal(), zero
  );
  return success();
}

LogicalResult AssertOpLowering::matchAndRewrite(
    AssertOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<boolean::AssertOp>(op, adaptor.getCond(), rewriter.getStringAttr(""));
  return success();
}

LogicalResult LowerVarArgsOp::matchAndRewrite(
    VarArgsOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  assert(adaptor.getElements().size() <= std::numeric_limits<int64_t>::max());
  rewriter.replaceOpWithNewOp<array::CreateArrayOp>(
      op,
      array::ArrayType::get(
          getTypeConverter()->convertType(
              llvm::cast<VarArgsType>(op.getResult().getType()).getInner()
          ),
          {static_cast<int64_t>(adaptor.getElements().size())}
      ),
      adaptor.getElements()
  );
  return success();
}

LogicalResult LowerReadBackOp::matchAndRewrite(
    ReadBackOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  const auto *converter = getTypeConverter();
  assert(converter);

  auto replace = [&](auto... args) {
    auto outType = converter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<component::FieldReadOp>(
        op, outType, adaptor.getComp(), adaptor.getFieldAttr(),
        std::forward<decltype(args)>(args)...
    );
  };
  llvm::TypeSwitch<Attribute>(adaptor.getDistance())
      .Case([&](SymbolRefAttr symAttr) {
    // If the distance is a symbol create an affine expression that negates the symbol.
    // Expects the symbol to be one of the parameters of the current struct. Will generate malformed
    // IR otherwise.
    replace(
        AffineMapAttr::get(AffineMap::get(
            /*dimCount=*/0, /*symbolCount=*/1,
            rewriter.getAffineConstantExpr(0) - rewriter.getAffineSymbolExpr(0)
        )),
        // materializeParam's Value return will be implicitly cast to a ValueRange.
        // The previous creation of a local variable `ValueRange mapOperands({value})`
        // causes errors as ValueRange is a thin wrapper.
        materializeParam(symAttr, rewriter, op->getLoc()), 0
    );
  })
      .Case([&](IntegerAttr intAttr) {
    // If the distance is a literal integer just negate it to move backwards.
    auto intValue = intAttr.getValue();
    assert(intValue.isNonNegative());
    if (!intValue.isZero()) {
      intValue.negate();
    }
    replace(IntegerAttr::get(rewriter.getIndexType(), intValue));
  })
      .Case([&](ConstExprAttr cexpAttr) {
    // If the distance is an expression, wrap it around a negation.
    auto templMap = AffineMap::get(
        /*dimCount=*/1, /*symbolCount=*/0,
        rewriter.getAffineConstantExpr(0) - rewriter.getAffineDimExpr(0)
    );
    auto newMap = templMap.compose(cexpAttr.getMap());
    auto newCexpr = ConstExprAttr::get(newMap, cexpAttr.getFormals());
    auto params = op->getParentOfType<component::StructDefOp>()
                      .getConstParams()
                      .value_or(rewriter.getArrayAttr({}))
                      .getValue();
    SmallVector<SmallVector<Value>, 1> mapOperandsMem(1);
    SmallVector<ValueRange, 1> mapOperands;
    materializeValuesForParams(
        {newCexpr}, params, MutableArrayRef(mapOperandsMem), rewriter, op.getLoc(), mapOperands
    );
    replace(Attribute(AffineMapAttr::get(newCexpr.getMap())), mapOperands[0], 0);
  }).Default([&](Attribute) { llvm_unreachable("ReadBackOp with an unexpected attribute type"); });
  return success();
}
LogicalResult LowerGlobalDefOp::matchAndRewrite(
    GlobalDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  Type newType = getTypeConverter()->convertType(op.getType());
  rewriter.replaceOpWithNewOp<global::GlobalDefOp>(op, op.getSymName(), false, newType, nullptr);
  return success();
}

LogicalResult LowerSetGlobalOp::matchAndRewrite(
    SetGlobalOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<global::GlobalWriteOp>(op, op.getNameRefAttr(), adaptor.getVal());
  return success();
}

LogicalResult LowerGetGlobalOp::matchAndRewrite(
    GetGlobalOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  Type newType = getTypeConverter()->convertType(op.getResult().getType());
  rewriter.replaceOpWithNewOp<global::GlobalReadOp>(op, newType, op.getNameRef());
  return success();
}

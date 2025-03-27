#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <llzk/Dialect/LLZK/IR/Types.h>
#include <llzk/Dialect/LLZK/Util/AttributeHelper.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
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

///////////////////////////////////////////////////////////
/// ZmirLitValOpLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult LitValOpLowering::matchAndRewrite(
    LitValOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::FeltConstantOp>(
      op, llzk::FeltConstAttr::get(getContext(), llvm::APInt(64, adaptor.getValue()))
  );
  return mlir::success();
}

LogicalResult LitStrOpLowering::matchAndRewrite(
    LitStrOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::LitStringOp>(op, adaptor.getValue());
  return success();
}

mlir::LogicalResult ComponentLowering::matchAndRewrite(
    SplitComponentOp op, OpAdaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto newOp = rewriter.create<llzk::StructDefOp>(
      op.getLoc(), op.getNameAttr(),
      op.getParams().has_value() ? *op.getParams() : rewriter.getArrayAttr({})
  );
  rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(), newOp.getRegion().end());
  rewriter.eraseOp(op);
  return success();
}

mlir::LogicalResult FieldDefOpLowering::matchAndRewrite(
    FieldDefOp op, OpAdaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::FieldDefOp>(
      op, op.getNameAttr(), getTypeConverter()->convertType(op.getType())
  );
  return mlir::success();
}

/// Inspired by FuncOp::cloneInto
void cloneAttrsIntoLlzkFunc(mlir::func::FuncOp src, llzk::FuncOp dest) {
  // Add the attributes of this function to dest (except visibility unless is
  // extern).
  llvm::MapVector<mlir::StringAttr, mlir::Attribute> newAttrMap;
  for (const auto &attr : dest->getAttrs()) {
    newAttrMap.insert({attr.getName(), attr.getValue()});
  }
  for (const auto &attr : src->getAttrs()) {
    newAttrMap.insert({attr.getName(), attr.getValue()});
  }

  if (!newAttrMap.contains(mlir::StringAttr::get(src.getContext(), "extern"))) {
    newAttrMap.erase(mlir::StringAttr::get(src.getContext(), "sym_visibility"));
  }

  auto newAttrs = llvm::to_vector(llvm::map_range(
      newAttrMap, [](std::pair<mlir::StringAttr, mlir::Attribute> attrPair
                  ) { return mlir::NamedAttribute(attrPair.first, attrPair.second); }
  ));
  dest->setAttrs(mlir::DictionaryAttr::get(src.getContext(), newAttrs));
}

mlir::LogicalResult FuncOpLowering::matchAndRewrite(
    mlir::func::FuncOp op, OpAdaptor, mlir::ConversionPatternRewriter &rewriter
) const {

  auto type = op.getFunctionType();
  mlir::TypeConverter::SignatureConversion result(type.getNumInputs());

  llvm::SmallVector<mlir::Type, 1> newResults;
  if (failed(getTypeConverter()->convertSignatureArgs(type.getInputs(), result)) ||
      failed(getTypeConverter()->convertTypes(type.getResults(), newResults)) ||
      failed(rewriter.convertRegionTypes(&op.getFunctionBody(), *getTypeConverter(), &result))) {
    return mlir::failure();
  }

  auto newType =
      mlir::FunctionType::get(rewriter.getContext(), result.getConvertedTypes(), newResults);

  auto newFuncOp = rewriter.create<llzk::FuncOp>(op.getLoc(), op.getNameAttr(), newType);
  cloneAttrsIntoLlzkFunc(op, newFuncOp);
  rewriter.inlineRegionBefore(op.getRegion(), newFuncOp.getRegion(), newFuncOp.end());
  rewriter.replaceOp(op, newFuncOp);
  return mlir::success();
}

mlir::LogicalResult ReturnOpLowering::matchAndRewrite(
    mlir::func::ReturnOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::ReturnOp>(op, adaptor.getOperands());
  return mlir::success();
}

mlir::LogicalResult ExternCallOpLowering::matchAndRewrite(
    mlir::func::CallIndirectOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto calleeOp = adaptor.getCallee().getDefiningOp();
  if (!calleeOp) {
    return failure();
  }
  auto callee = dyn_cast<ExternFnRefOp>(calleeOp);
  if (!callee) {
    return failure();
  }

  llvm::SmallVector<mlir::Type> results;

  auto convRes = getTypeConverter()->convertTypes(op.getCallee().getType().getResults(), results);
  if (mlir::failed(convRes)) {
    return op->emitError("failed to transform zml types into llzk types");
  }
  rewriter.replaceOpWithNewOp<llzk::CallOp>(
      op, results, callee.getNameAttr(), adaptor.getCalleeOperands()
  );
  return mlir::success();
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
  auto type = mlir::cast<llzk::ArrayType>(src.getType());
  if (type.getDimensionSizes().size() == 1) {
    return builder.create<llzk::ReadArrayOp>(loc, src, iv);
  }
  return builder.create<llzk::ExtractArrayOp>(loc, src, iv);
}

static void writeArray(Value dst, Value iv, Value val, OpBuilder &builder, Location loc) {
  auto type = mlir::cast<llzk::ArrayType>(dst.getType());
  if (type.getDimensionSizes().size() == 1) {
    builder.create<llzk::WriteArrayOp>(loc, dst, iv, val);
  } else {
    builder.create<llzk::InsertArrayOp>(loc, dst, iv, val);
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
  auto targetInner = mlir::cast<ComponentType>(target).getArrayInnerType();
  auto targetArrayType = mlir::cast<llzk::ArrayType>(tc.convertType(target));

  auto array = builder.create<llzk::CreateArrayOp>(loc, targetArrayType);

  auto lb = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto ub = builder.create<llzk::ArrayLengthOp>(
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

static Value readSuperFields(
    ComponentType type, Value chain, Type target, const TypeConverter &tc, Location loc,
    OpBuilder &builder
) {

  auto superComp = mlir::dyn_cast_if_present<ComponentType>(type.getSuperType());
  // Stop if we reached a builtin that transforms to a primitive llzk type, we are done, or we
  // cannot continue extracting
  if (type == target || wasConvertedToPrimitiveType(type) || !superComp ||
      mlir::isa<TypeVarType>(type.getSuperType())) {
    return chain;
  }
  if (type.isConcreteArray()) {
    auto targetAsComp = mlir::dyn_cast_if_present<ComponentType>(target);
    // If the target is an array component then read the super fields of the inner elements and
    // returns a new array with those values.
    if (targetAsComp && targetAsComp.isConcreteArray()) {
      return copyArraySuperFields(
          mlir::cast<ComponentType>(*type.getArrayInnerType()), chain, target, tc, loc, builder
      );
    }
    return chain;
  }

  auto read = builder.create<llzk::FieldReadOp>(loc, tc.convertType(superComp), chain, "$super");
  return readSuperFields(superComp, read, target, tc, loc, builder);
}

LogicalResult LowerSuperCoerceOp::matchAndRewrite(
    SuperCoerceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto t = mlir::dyn_cast<ComponentType>(op.getOperand().getType());
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
  auto comp = mlir::dyn_cast<ComponentType>(compType);
  if (!comp) {
    return op->emitOpError() << "was expecting a component type but got " << compType;
  }
  auto compName = comp.getName().getAttr();
  if (wasConvertedToPrimitiveType(comp)) {
    rewriter.eraseOp(op);
    return success();
  }
  auto sym = mlir::SymbolRefAttr::get(
      compName, {mlir::SymbolRefAttr::get(rewriter.getStringAttr("constrain"))}
  );
  rewriter.replaceOpWithNewOp<llzk::CallOp>(op, TypeRange(), sym, adaptor.getOperands());

  return success();
}

LogicalResult LowerLoadValParamOp::matchAndRewrite(
    LoadValParamOp op, OpAdaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::ConstReadOp>(
      op, llzk::FeltType::get(getContext()), mlir::SymbolRefAttr::get(op.getParamAttr())
  );
  return success();
}

/// Given an attribute materializes it into a Value if it's either a SymbolRefAttr or an
/// IntegerAttr. Any other kind of Attribute is considered malformed IR and will abort.
static Value materializeParam(Attribute attr, OpBuilder &builder, Location loc) {
  if (auto symAttr = mlir::dyn_cast<SymbolRefAttr>(attr)) {
    auto param = builder.create<llzk::ConstReadOp>(
        loc, llzk::FeltType::get(builder.getContext()), symAttr.getRootReference()
    );
    return builder.create<llzk::FeltToIndexOp>(loc, param);
  }
  if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
    return builder.create<arith::ConstantIndexOp>(loc, llzk::fromAPInt(intAttr.getValue()));
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
    if (auto param = mlir::dyn_cast<ConstExprAttr>(attr)) {
      out.push_back(param);
    }
  }
}

static void materializeValuesForParams(
    ArrayRef<ConstExprAttr> affineParams, ArrayRef<Attribute> sourceParams,
    MutableArrayRef<SmallVector<Value>> mem, OpBuilder &builder, Location loc,
    SmallVectorImpl<ValueRange> &out
) {
  auto res = llvm::map_to_vector(llvm::zip_equal(mem, affineParams), [&](auto in) -> ValueRange {
    auto [values, param] = in;
    values = llvm::map_to_vector(param.getFormals(), [&](auto formal) {
      return materializeParam(sourceParams[formal], builder, loc);
    });

    return values;
  });
  out.insert(out.end(), res.begin(), res.end());
}

mlir::LogicalResult CallIndirectOpLoweringInCompute::matchAndRewrite(
    mlir::func::CallIndirectOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto parent = op->getParentOfType<llzk::FuncOp>();
  if (!parent || parent.getName() != "compute") {
    return failure(); // Don't operate on non compute calls
  }

  auto callee = mlir::dyn_cast<ConstructorRefOp>(adaptor.getCallee().getDefiningOp());
  if (!callee) {
    return failure();
  }

  Params params{
      .callee = mlir::cast<ComponentType>(op.getResult(0).getType()).getParams(),
      .caller = op->getParentOfType<llzk::StructDefOp>()
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

  auto sym = mlir::SymbolRefAttr::get(
      callee.getComponentAttr().getAttr(), {mlir::SymbolRefAttr::get(parent.getNameAttr())}
  );

  llvm::SmallVector<mlir::Type> types;
  auto convRes = getTypeConverter()->convertTypes(op.getResultTypes(), types);
  if (mlir::failed(convRes)) {
    return op->emitError("failed to transform zml types into llzk types");
  }
  mlir::ValueRange args(
      mlir::iterator_range(adaptor.getOperands().begin() + 1, adaptor.getOperands().end())
  );

  rewriter.replaceOpWithNewOp<llzk::CallOp>(op, types, sym, mapOperands, dimsPerMap, args);
  return mlir::success();
}

mlir::LogicalResult WriteFieldOpLowering::matchAndRewrite(
    WriteFieldOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::FieldWriteOp>(
      op, adaptor.getComponent(), adaptor.getFieldNameAttr(), adaptor.getVal()
  );
  return mlir::success();
}

mlir::LogicalResult RemoveConstructorRefOp::matchAndRewrite(
    ConstructorRefOp op, OpAdaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.eraseOp(op);
  return mlir::success();
}

mlir::LogicalResult RemoveExternFnRefOp::matchAndRewrite(
    ExternFnRefOp op, OpAdaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.eraseOp(op);
  return mlir::success();
}

mlir::LogicalResult LowerReadFieldOp::matchAndRewrite(
    ReadFieldOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::FieldReadOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getComponent(),
      adaptor.getFieldNameAttr()
  );
  return mlir::success();
}

mlir::LogicalResult LowerConstrainOp::matchAndRewrite(
    ConstrainOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::EmitEqualityOp>(op, adaptor.getLhs(), adaptor.getRhs());
  return mlir::success();
}

mlir::LogicalResult LowerInRangeOp::matchAndRewrite(
    InRangeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto le = rewriter.create<llzk::CmpOp>(
      op.getLoc(), llzk::FeltCmpPredicateAttr::get(getContext(), llzk::FeltCmpPredicate::LE),
      adaptor.getLow(), adaptor.getMid()
  );
  auto lt = rewriter.create<llzk::CmpOp>(
      op.getLoc(), llzk::FeltCmpPredicateAttr::get(getContext(), llzk::FeltCmpPredicate::LT),
      adaptor.getMid(), adaptor.getHigh()
  );
  auto convLe = rewriter.create<llzk::IntToFeltOp>(op.getLoc(), le);
  auto convLt = rewriter.create<llzk::IntToFeltOp>(op.getLoc(), lt);
  auto mul = rewriter.create<llzk::MulFeltOp>(op.getLoc(), convLe, convLt);
  rewriter.replaceOp(op, mul);

  return mlir::success();
}

mlir::LogicalResult LowerNewArrayOp::matchAndRewrite(
    NewArrayOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  // NewArrayOp is SameTypeOperands so by querying the type of any one of the elements we can know
  // if the inputs are scalar or arrays

  auto type = getTypeConverter()->convertType(op.getType());
  auto arrType = dyn_cast<llzk::ArrayType>(type);
  if (!arrType) {
    return op.emitOpError() << "was expecting an array type";
  }

  // If it's an array then we allocate an empty one and then insert each operand with InsertArrayOp
  if (!adaptor.getElements().empty() && isa<llzk::ArrayType>(adaptor.getElements()[0].getType())) {
    Location cachedLoc = op.getLoc();
    auto arr = rewriter.replaceOpWithNewOp<llzk::CreateArrayOp>(op, arrType, ValueRange());
    for (size_t i = 0; i < adaptor.getElements().size(); i++) {
      auto idx = rewriter.create<arith::ConstantIndexOp>(cachedLoc, i);
      rewriter.create<llzk::InsertArrayOp>(
          cachedLoc, arr, ValueRange({idx}), adaptor.getElements()[i]
      );
    }
  } else {
    rewriter.replaceOpWithNewOp<llzk::CreateArrayOp>(op, arrType, adaptor.getElements());
  }
  return mlir::success();
}

mlir::LogicalResult LowerLitValArrayOp::matchAndRewrite(
    LitValArrayOp op, OpAdaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto type = getTypeConverter()->convertType(op.getType());
  auto arrType = dyn_cast<llzk::ArrayType>(type);
  if (!arrType) {
    return op.emitOpError() << "was expecting an array type";
  }

  SmallVector<Value> lits;
  llzk::FeltType felt = llzk::FeltType::get(getContext());
  std::transform(
      op.getElements().begin(), op.getElements().end(), std::back_inserter(lits),
      [&](int64_t value) {
    return rewriter.create<llzk::FeltConstantOp>(
        op.getLoc(), felt, llzk::FeltConstAttr::get(getContext(), llzk::toAPInt(value))
    );
  }
  );

  rewriter.replaceOpWithNewOp<llzk::CreateArrayOp>(op, arrType, ValueRange(lits));
  return success();
}

mlir::LogicalResult LowerReadArrayOp::matchAndRewrite(
    ReadArrayOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  llvm::SmallVector<mlir::Value> toIndexOps;
  std::transform(
      adaptor.getIndices().begin(), adaptor.getIndices().end(), std::back_inserter(toIndexOps),
      [&](mlir::Value index) -> mlir::TypedValue<mlir::IndexType> {
    if (mlir::isa<mlir::IndexType>(index.getType())) {
      return mlir::cast<mlir::TypedValue<mlir::IndexType>>(index);
    }
    return rewriter.create<llzk::FeltToIndexOp>(op.getLoc(), index).getResult();
  }
  );
  auto convertedType = getTypeConverter()->convertType(op.getType());
  if (isa<llzk::ArrayType>(convertedType)) {
    rewriter.replaceOpWithNewOp<llzk::ExtractArrayOp>(
        op, convertedType, adaptor.getLvalue(), toIndexOps
    );
  } else {
    rewriter.replaceOpWithNewOp<llzk::ReadArrayOp>(
        op, convertedType, adaptor.getLvalue(), toIndexOps
    );
  }
  return mlir::success();
}

mlir::LogicalResult LowerIsz::matchAndRewrite(
    IsZeroOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto zero = rewriter.create<llzk::FeltConstantOp>(
      op.getLoc(), llzk::FeltConstAttr::get(getContext(), mlir::APInt::getZero(64))
  );
  auto cmpOp = rewriter.create<llzk::CmpOp>(
      op.getLoc(), llzk::FeltCmpPredicateAttr::get(getContext(), llzk::FeltCmpPredicate::EQ),
      adaptor.getIn(), zero
  );
  rewriter.replaceOpWithNewOp<llzk::IntToFeltOp>(op, cmpOp);
  return mlir::success();
}

mlir::LogicalResult LowerAllocArrayOp::matchAndRewrite(
    AllocArrayOp op, OpAdaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto type = getTypeConverter()->convertType(op.getType());
  auto arrType = dyn_cast<llzk::ArrayType>(type);
  if (!arrType) {
    return op.emitOpError() << "was expecting an array type";
  }

  ArrayAttr compParams = op->getParentOfType<llzk::StructDefOp>().getType().getParams();
  SmallVector<ConstExprAttr> affineArrayParams;
  ComponentType cType = op.getResult().getType();
  for (Attribute attr : cType.getParams()) {
    if (auto param = mlir::dyn_cast<ConstExprAttr>(attr)) {
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
      assert(
          static_cast<unsigned int>(formal) < compParams.size() &&
          "Can only use as map operands declared parameters"
      );
      values.push_back(
          materializeParam(compParams[static_cast<unsigned int>(formal)], rewriter, op->getLoc())
      );
    }
    mapOperands.push_back(values);
  }

  rewriter.replaceOpWithNewOp<llzk::CreateArrayOp>(op, arrType, mapOperands, dimsPerMap);

  return mlir::success();
}

mlir::LogicalResult LowerArrayLengthOp::matchAndRewrite(
    GetArrayLenOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto c = rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), mlir::IndexType::get(getContext()), rewriter.getIndexAttr(0)
  );
  rewriter.replaceOpWithNewOp<llzk::ArrayLengthOp>(op, adaptor.getArray(), c);

  return mlir::success();
}

mlir::LogicalResult LowerIndexToValOp::matchAndRewrite(
    IndexToValOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::IntToFeltOp>(
      op, llzk::FeltType::get(getContext()), adaptor.getIndex()
  );

  return mlir::success();
}

mlir::LogicalResult LowerValToIndexOp::matchAndRewrite(
    ValToIndexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  if (mlir::isa<mlir::IndexType>(adaptor.getVal().getType())) {
    rewriter.replaceAllUsesWith(op, adaptor.getVal());
    rewriter.eraseOp(op);
  } else {
    rewriter.replaceOpWithNewOp<llzk::FeltToIndexOp>(op, adaptor.getVal());
  }

  return mlir::success();
}

mlir::LogicalResult LowerWriteArrayOp::matchAndRewrite(
    WriteArrayOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  if (mlir::isa<llzk::ArrayType>(adaptor.getValue().getType())) {
    rewriter.replaceOpWithNewOp<llzk::InsertArrayOp>(
        op, adaptor.getArray(), adaptor.getIndices(), adaptor.getValue()
    );
  } else {
    rewriter.replaceOpWithNewOp<llzk::WriteArrayOp>(
        op, adaptor.getArray(), adaptor.getIndices(), adaptor.getValue()
    );
  }

  return mlir::success();
}

mlir::LogicalResult UpdateScfExecuteRegionOpTypes::matchAndRewrite(
    mlir::scf::ExecuteRegionOp op, OpAdaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  SmallVector<Type> newTypes;
  if (mlir::failed(getTypeConverter()->convertTypes(op.getResultTypes(), newTypes))) {
    return mlir::failure();
  }
  auto exec = rewriter.create<mlir::scf::ExecuteRegionOp>(op.getLoc(), newTypes);
  rewriter.inlineRegionBefore(op.getRegion(), exec.getRegion(), exec.getRegion().end());
  rewriter.replaceOp(op, exec);

  return mlir::success();
}

LogicalResult ValToI1OpLowering::matchAndRewrite(
    ValToI1Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto zero = rewriter.create<llzk::FeltConstantOp>(
      op.getLoc(), llzk::FeltType::get(getContext()),
      llzk::FeltConstAttr::get(getContext(), APInt(1, 0))
  );
  rewriter.replaceOpWithNewOp<llzk::CmpOp>(
      op, llzk::FeltCmpPredicateAttr::get(getContext(), llzk::FeltCmpPredicate::NE),
      adaptor.getVal(), zero
  );
  return success();
}

LogicalResult AssertOpLowering::matchAndRewrite(
    AssertOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::AssertOp>(op, adaptor.getCond(), rewriter.getStringAttr(""));
  return success();
}

LogicalResult LowerVarArgsOp::matchAndRewrite(
    VarArgsOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  assert(adaptor.getElements().size() <= std::numeric_limits<int64_t>::max());
  rewriter.replaceOpWithNewOp<llzk::CreateArrayOp>(
      op,
      llzk::ArrayType::get(
          getTypeConverter()->convertType(
              mlir::cast<VarArgsType>(op.getResult().getType()).getInner()
          ),
          {static_cast<int64_t>(adaptor.getElements().size())}
      ),
      adaptor.getElements()
  );
  return success();
}

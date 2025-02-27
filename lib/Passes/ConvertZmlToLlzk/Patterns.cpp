#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <llzk/Dialect/LLZK/IR/Types.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <unordered_set>
#include <vector>
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
  llzk::FeltType felt = llzk::FeltType::get(getContext());
  rewriter.replaceOpWithNewOp<llzk::FeltConstantOp>(
      op, felt, llzk::FeltConstAttr::get(getContext(), llvm::APInt(64, adaptor.getValue()))
  );
  return mlir::success();
}

mlir::LogicalResult ComponentLowering::matchAndRewrite(
    SplitComponentOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
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
    FieldDefOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
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
    mlir::func::FuncOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {

  auto type = op.getFunctionType();
  mlir::TypeConverter::SignatureConversion result(op.getFunctionType().getInputs().size());

  llvm::SmallVector<mlir::Type, 1> newResults;
  if (failed(getTypeConverter()->convertSignatureArgs(type.getInputs(), result)) ||
      failed(getTypeConverter()->convertTypes(type.getResults(), newResults)) ||
      failed(rewriter.convertRegionTypes(&op.getFunctionBody(), *getTypeConverter(), &result))) {
    return mlir::failure();
  }

  auto newType =
      mlir::FunctionType::get(rewriter.getContext(), result.getConvertedTypes(), newResults);

  auto newFuncOp = rewriter.replaceOpWithNewOp<llzk::FuncOp>(op, op.getNameAttr(), newType);
  cloneAttrsIntoLlzkFunc(op, newFuncOp);

  rewriter.inlineRegionBefore(op.getRegion(), newFuncOp.getRegion(), newFuncOp.end());
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

// Set of the builtins (by name) that are converted to a LLZK felt. Used for shortcircuiting the
// lowering of SuperCoerceOp and for removing the calls to @constrain that point to structs that get
// removed during lowering because the implementation of these types gets removed.
static std::unordered_set<std::string_view> feltEquivalentTypes{"Val",    "Add", "Sub", "Mul",
                                                                "BitAnd", "Inv", "Isz", "InRange"};

LogicalResult LowerSuperCoerceOp::matchAndRewrite(
    SuperCoerceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto t = mlir::dyn_cast<ComponentType>(op.getOperand().getType());
  auto opChain = adaptor.getComponent();
  auto tc = getTypeConverter();
  while (t != op.getResult().getType()) {
    if (t.getBuiltin() &&
        feltEquivalentTypes.find(t.getName().getValue()) != feltEquivalentTypes.end()) {
      // This type will get converted to a felt so there is not need to extract the super value any
      // longer.
      break;
    }
    if (t.getName().getValue() == "Array") {
      break;
    }
    auto typ = t.getSuperType();
    if (!typ) {
      break;
    }
    if (auto comp = mlir::dyn_cast<ComponentType>(typ)) {
      t = comp;
    }
    if (auto comp = mlir::dyn_cast<TypeVarType>(typ)) {
      break;
    }
    opChain =
        rewriter.create<llzk::FieldReadOp>(op.getLoc(), tc->convertType(t), opChain, "$super");
    assert(t);
  }
  rewriter.replaceOp(op, opChain);
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
  if (comp.getBuiltin() &&
      feltEquivalentTypes.find(compName.getValue()) != feltEquivalentTypes.end()) {
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
    LoadValParamOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::ConstReadOp>(
      op, llzk::FeltType::get(getContext()), mlir::SymbolRefAttr::get(op.getParamAttr())
  );
  return success();
}

mlir::LogicalResult CallIndirectOpLoweringInCompute::matchAndRewrite(
    mlir::func::CallIndirectOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto parent = op->getParentOfType<llzk::FuncOp>();
  if (!parent || parent.getName() != "compute") {
    return mlir::failure(); // Don't operate on non compute calls
  }

  auto callee = mlir::dyn_cast<ConstructorRefOp>(adaptor.getCallee().getDefiningOp());
  if (!callee) {
    return failure();
  }
  auto comp = callee.getComponentAttr();

  auto sym =
      mlir::SymbolRefAttr::get(comp.getAttr(), {mlir::SymbolRefAttr::get(parent.getNameAttr())});

  llvm::SmallVector<mlir::Type> types;
  auto convRes = getTypeConverter()->convertTypes(op.getResultTypes(), types);
  if (mlir::failed(convRes)) {
    return op->emitError("failed to transform zml types into llzk types");
  }
  mlir::ValueRange args(
      mlir::iterator_range(adaptor.getOperands().begin() + 1, adaptor.getOperands().end())
  );

  rewriter.replaceOpWithNewOp<llzk::CallOp>(op, types, sym, args);
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
    ConstructorRefOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.eraseOp(op);
  return mlir::success();
}

mlir::LogicalResult RemoveExternFnRefOp::matchAndRewrite(
    ExternFnRefOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
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
    auto arr = rewriter.replaceOpWithNewOp<llzk::CreateArrayOp>(op, arrType, ValueRange());
    for (size_t i = 0; i < adaptor.getElements().size(); i++) {
      auto idx = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), i);
      rewriter.create<llzk::InsertArrayOp>(
          op.getLoc(), arr, ValueRange({idx}), adaptor.getElements()[i]
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
      [&](long value) {
    return rewriter.create<llzk::FeltConstantOp>(
        op.getLoc(), felt, llzk::FeltConstAttr::get(getContext(), llvm::APInt(64, value))
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
    AllocArrayOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto type = getTypeConverter()->convertType(op.getType());
  auto arrType = dyn_cast<llzk::ArrayType>(type);
  if (!arrType) {
    return op.emitOpError() << "was expecting an array type";
  }
  rewriter.replaceOpWithNewOp<llzk::CreateArrayOp>(op, arrType, mlir::ValueRange());

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
    mlir::scf::ExecuteRegionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  SmallVector<Type> newTypes;
  if (mlir::failed(getTypeConverter()->convertTypes(op.getResultTypes(), newTypes))) {
    return mlir::failure();
  }
  auto exec = rewriter.replaceOpWithNewOp<mlir::scf::ExecuteRegionOp>(op, newTypes);
  rewriter.inlineRegionBefore(op.getRegion(), exec.getRegion(), exec.getRegion().end());

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

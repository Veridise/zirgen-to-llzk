#include "zklang/Passes/ConvertZmlToLlzk/Patterns.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include "zklang/Dialect/ZML/IR/Types.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <unordered_set>
#include <vector>

using namespace zkc;
using namespace mlir;

///////////////////////////////////////////////////////////
/// ZmirLitValOpLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult Zmir::LitValOpLowering::matchAndRewrite(
    Zmir::LitValOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  llzk::FeltType felt = llzk::FeltType::get(getContext());
  rewriter.replaceOpWithNewOp<llzk::FeltConstantOp>(
      op, felt, llzk::FeltConstAttr::get(getContext(), llvm::APInt(64, adaptor.getValue()))
  );
  return mlir::success();
}

mlir::LogicalResult Zmir::ComponentLowering::matchAndRewrite(
    Zmir::SplitComponentOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto newOp = rewriter.replaceOpWithNewOp<llzk::StructDefOp>(
      op, op.getNameAttr(), op.getParams().has_value() ? *op.getParams() : rewriter.getArrayAttr({})
  );
  {
    mlir::OpBuilder::InsertionGuard insertionGuard(rewriter);
    auto *block = rewriter.createBlock(&newOp.getRegion());
    rewriter.setInsertionPointToStart(block);

    // Copy the field definitions
    for (auto paramOp : op.getOps<FieldDefOp>()) {
      rewriter.clone(*paramOp.getOperation());
    }
    {
      mlir::OpBuilder::InsertionGuard modGuard(rewriter);
      rewriter.setInsertionPointAfter(newOp);
      for (auto funcOp : op.getOps<mlir::func::FuncOp>()) {
        if (funcOp == op.getBodyFunc() || funcOp == op.getConstrainFunc()) {
          continue;
        }

        rewriter.clone(*funcOp.getOperation());
      }
    }
    rewriter.clone(*op.getBodyFunc());
    rewriter.clone(*op.getConstrainFunc());
  }
  return mlir::success();
}

mlir::LogicalResult Zmir::FieldDefOpLowering::matchAndRewrite(
    Zmir::FieldDefOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
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

mlir::LogicalResult Zmir::FuncOpLowering::matchAndRewrite(
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
  newFuncOp.getRegion().takeBody(op.getRegion());
  return mlir::success();
}

mlir::LogicalResult Zmir::ReturnOpLowering::matchAndRewrite(
    mlir::func::ReturnOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::ReturnOp>(op, adaptor.getOperands());
  return mlir::success();
}

mlir::LogicalResult Zmir::CallOpLowering::matchAndRewrite(
    mlir::func::CallOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  llvm::SmallVector<mlir::Type> results;

  auto convRes = getTypeConverter()->convertTypes(op.getCalleeType().getResults(), results);
  if (mlir::failed(convRes)) {
    return op->emitError("failed to transform zml types into llzk types");
  }
  rewriter.replaceOpWithNewOp<llzk::CallOp>(
      op, adaptor.getCallee(), results, adaptor.getOperands()
  );
  return mlir::success();
}

LogicalResult Zmir::LowerNopOp::matchAndRewrite(
    Zmir::NopOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  if (op.getNumOperands() == op.getNumResults()) {
    rewriter.replaceAllUsesWith(op.getResults(), adaptor.getOperands());
  }
  rewriter.eraseOp(op);
  return success();
}

static std::unordered_set<std::string_view> feltEquivalentTypes{"Val",    "Add", "Sub", "Mul",
                                                                "BitAnd", "Inv", "Isz"};

LogicalResult Zmir::LowerSuperCoerceOp::matchAndRewrite(
    Zmir::SuperCoerceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto t = mlir::dyn_cast<Zmir::ComponentType>(op.getOperand().getType());
  auto opChain = adaptor.getComponent();
  auto tc = getTypeConverter();
  while (t != op.getResult().getType()) {
    if (feltEquivalentTypes.find(t.getName().getValue()) != feltEquivalentTypes.end() &&
        t.getBuiltin()) {
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
    if (auto comp = mlir::dyn_cast<Zmir::ComponentType>(typ)) {
      t = comp;
    }
    if (auto comp = mlir::dyn_cast<Zmir::TypeVarType>(typ)) {
      break;
    }
    opChain =
        rewriter.create<llzk::FieldReadOp>(op.getLoc(), tc->convertType(t), opChain, "$super");
    assert(t);
  }
  rewriter.replaceOp(op, opChain);
  return success();
}

LogicalResult Zmir::LowerConstrainCallOp::matchAndRewrite(
    Zmir::ConstrainCallOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto compType = op.getSelf().getType();
  auto comp = mlir::dyn_cast<Zmir::ComponentType>(compType);
  if (!comp) {
    return op->emitOpError() << "was expecting a component type but got " << compType;
  }
  auto compName = comp.getName().getAttr();
  if (feltEquivalentTypes.find(compName.getValue()) != feltEquivalentTypes.end()) {
    rewriter.eraseOp(op);
    return success();
  }
  auto sym = mlir::SymbolRefAttr::get(
      compName, {mlir::SymbolRefAttr::get(rewriter.getStringAttr("constrain"))}
  );
  rewriter.replaceOpWithNewOp<llzk::CallOp>(op, sym, TypeRange(), adaptor.getOperands());

  return success();
}

LogicalResult Zmir::LowerLoadValParamOp::matchAndRewrite(
    Zmir::LoadValParamOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::ConstReadOp>(
      op, llzk::FeltType::get(getContext()), mlir::SymbolRefAttr::get(op.getParamAttr())
  );
  return success();
}

mlir::LogicalResult Zmir::CallIndirectOpLoweringInCompute::matchAndRewrite(
    mlir::func::CallIndirectOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto parent = op->getParentOfType<llzk::FuncOp>();
  if (!parent || parent.getName() != "compute") {
    return mlir::failure(); // Don't operate on non compute calls
  }

  auto callee = mlir::dyn_cast<Zmir::ConstructorRefOp>(adaptor.getCallee().getDefiningOp());
  if (!callee) {
    return op->emitOpError() << "was expecting the callee comes from an zmir.constructor op";
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

  rewriter.replaceOpWithNewOp<llzk::CallOp>(op, sym, types, args);
  return mlir::success();
}

mlir::LogicalResult Zmir::WriteFieldOpLowering::matchAndRewrite(
    Zmir::WriteFieldOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::FieldWriteOp>(
      op, adaptor.getComponent(), adaptor.getFieldNameAttr(), adaptor.getVal()
  );
  return mlir::success();
}

mlir::LogicalResult Zmir::RemoveConstructorRefOp::matchAndRewrite(
    Zmir::ConstructorRefOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.eraseOp(op);
  return mlir::success();
}

mlir::LogicalResult Zmir::LowerReadFieldOp::matchAndRewrite(
    Zmir::ReadFieldOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::FieldReadOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getComponent(),
      adaptor.getFieldNameAttr()
  );
  return mlir::success();
}

mlir::LogicalResult Zmir::LowerConstrainOp::matchAndRewrite(
    Zmir::ConstrainOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::EmitEqualityOp>(op, adaptor.getLhs(), adaptor.getRhs());
  return mlir::success();
}

mlir::LogicalResult Zmir::LowerInRangeOp::matchAndRewrite(
    Zmir::InRangeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto le = rewriter.create<llzk::CmpOp>(
      op.getLoc(), llzk::FeltCmpPredicateAttr::get(getContext(), llzk::FeltCmpPredicate::LE),
      adaptor.getLow(), adaptor.getMid()
  );
  auto lt = rewriter.create<llzk::CmpOp>(
      op.getLoc(), llzk::FeltCmpPredicateAttr::get(getContext(), llzk::FeltCmpPredicate::LT),
      adaptor.getMid(), adaptor.getHigh()
  );
  auto mul = rewriter.create<mlir::arith::MulIOp>(op.getLoc(), le, lt);
  auto conv = rewriter.create<llzk::IntToFeltOp>(op.getLoc(), mul);
  rewriter.replaceOp(op, conv);

  return mlir::success();
}

mlir::LogicalResult Zmir::LowerNewArrayOp::matchAndRewrite(
    Zmir::NewArrayOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::CreateArrayOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getElements()
  );
  return mlir::success();
}

mlir::LogicalResult Zmir::LowerReadArrayOp::matchAndRewrite(
    Zmir::ReadArrayOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
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
  rewriter.replaceOpWithNewOp<llzk::ReadArrayOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getLvalue(), toIndexOps
  );
  return mlir::success();
}

mlir::LogicalResult Zmir::LowerIsz::matchAndRewrite(
    Zmir::IsZeroOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
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

mlir::LogicalResult Zmir::LowerAllocArrayOp::matchAndRewrite(
    Zmir::AllocArrayOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::CreateArrayOp>(
      op, getTypeConverter()->convertType(op.getType()), mlir::ValueRange()
  );

  return mlir::success();
}

mlir::LogicalResult Zmir::LowerArrayLengthOp::matchAndRewrite(
    Zmir::GetArrayLenOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto c = rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), mlir::IndexType::get(getContext()), rewriter.getIndexAttr(0)
  );
  rewriter.replaceOpWithNewOp<llzk::ArrayLengthOp>(op, adaptor.getArray(), c);

  return mlir::success();
}

mlir::LogicalResult Zmir::LowerIndexToValOp::matchAndRewrite(
    Zmir::IndexToValOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::IntToFeltOp>(
      op, llzk::FeltType::get(getContext()), adaptor.getIndex()
  );

  return mlir::success();
}

mlir::LogicalResult Zmir::LowerValToIndexOp::matchAndRewrite(
    Zmir::ValToIndexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  if (mlir::isa<mlir::IndexType>(adaptor.getVal().getType())) {
    rewriter.replaceAllUsesWith(op, adaptor.getVal());
    rewriter.eraseOp(op);
  } else {
    rewriter.replaceOpWithNewOp<llzk::FeltToIndexOp>(op, adaptor.getVal());
  }

  return mlir::success();
}

mlir::LogicalResult Zmir::LowerWriteArrayOp::matchAndRewrite(
    Zmir::WriteArrayOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::WriteArrayOp>(
      op, adaptor.getArray(), adaptor.getIndices(), adaptor.getValue()
  );

  return mlir::success();
}

mlir::LogicalResult Zmir::UpdateScfForOpTypes::matchAndRewrite(
    mlir::scf::ForOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {

  auto initArgs = adaptor.getInitArgs();
  rewriter.replaceOpWithNewOp<mlir::scf::ForOp>(
      op, adaptor.getLowerBound(), adaptor.getUpperBound(), adaptor.getStep(), initArgs,
      [&](mlir::OpBuilder &, mlir::Location loc, mlir::Value iv, mlir::ValueRange args) {
    mlir::SmallVector<mlir::Value> allArgs({iv}), finalArgs;

    allArgs.insert(allArgs.end(), args.begin(), args.end());
    for (auto [arg, origType] :
         llvm::zip_equal(allArgs, op.getRegion().front().getArgumentTypes())) {
      // To avoid redundant casts that cannot be reconciled only make a cast if the types difer
      if (arg.getType() == origType) {
        finalArgs.push_back(arg);
      } else {
        auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, origType, arg);
        finalArgs.push_back(cast.getResult(0));
      }
    }

    auto loopPrologue = rewriter.getInsertionBlock();

    rewriter.inlineBlockBefore(
        &op.getRegion().front(), loopPrologue, loopPrologue->end(), finalArgs
    );
  }
  );

  return mlir::success();
}

mlir::LogicalResult Zmir::UpdateScfYieldOpTypes::matchAndRewrite(
    mlir::scf::YieldOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {

  rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, adaptor.getOperands());

  return mlir::success();
}

LogicalResult Zmir::UpdateScfIfOpTypes::matchAndRewrite(
    scf::IfOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  SmallVector<Type, 1> newTypes;
  if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newTypes))) {
    return failure();
  }
  auto newIf =
      rewriter.replaceOpWithNewOp<scf::IfOp>(op, newTypes, adaptor.getCondition(), true, true);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    auto &thenBlock = newIf.getThenRegion().front();
    rewriter.inlineBlockBefore(&op.getThenRegion().front(), &thenBlock, thenBlock.end());
  }
  {
    OpBuilder::InsertionGuard guard(rewriter);
    auto &elseBlock = newIf.getElseRegion().front();
    rewriter.inlineBlockBefore(&op.getElseRegion().front(), &elseBlock, elseBlock.end());
  }
  return success();
}

mlir::LogicalResult Zmir::UpdateScfExecuteRegionOpTypes::matchAndRewrite(
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

LogicalResult Zmir::ValToI1OpLowering::matchAndRewrite(
    Zmir::ValToI1Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
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

LogicalResult Zmir::AssertOpLowering::matchAndRewrite(
    Zmir::AssertOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<llzk::AssertOp>(op, adaptor.getCond(), rewriter.getStringAttr(""));
  return success();
}

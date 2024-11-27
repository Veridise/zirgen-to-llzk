#include "Patterns.h"
#include "Helpers.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include "zkir/Dialect/ZKIR/IR/Ops.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>
#include <vector>

using namespace zkc;

///////////////////////////////////////////////////////////
/// ZmirLitValOpLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult Zmir::LitValOpLowering::matchAndRewrite(
    Zmir::LitValOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  zkir::FeltType felt = zkir::FeltType::get(getContext());
  rewriter.replaceOpWithNewOp<zkir::FeltConstantOp>(
      op, felt,
      zkir::FeltConstAttr::get(getContext(),
                               llvm::APInt(64, adaptor.getValue())));
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZmirGetSelfOpLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult Zmir::GetSelfOpLowering::matchAndRewrite(
    Zmir::GetSelfOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<zkir::CreateStructOp>(
      op, getTypeConverter()->convertType(op.getType()));
  return mlir::success();
}

mlir::LogicalResult Zmir::ComponentLowering::matchAndRewrite(
    Zmir::SplitComponentOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // TODO type parameters
  auto newOp = rewriter.replaceOpWithNewOp<zkir::StructDefOp>(
      op, op.getNameAttr(),
      op.getConstParams().has_value() ? *op.getConstParams()
                                      : rewriter.getArrayAttr({}));
  mlir::ModuleOp mod = op->getParentOfType<mlir::ModuleOp>();
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
        if (funcOp == op.getBodyFunc() || funcOp == op.getConstrainFunc())
          continue;

        rewriter.clone(*funcOp.getOperation());
      }
    }
    // TODO: Extra stuff in the component needs to go outside into the module
    rewriter.clone(*op.getBodyFunc());
    rewriter.clone(*op.getConstrainFunc());
  }
  return mlir::success();
}

mlir::LogicalResult Zmir::FieldDefOpLowering::matchAndRewrite(
    Zmir::FieldDefOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<zkir::FieldDefOp>(
      op, op.getNameAttr(), getTypeConverter()->convertType(op.getType()));
  return mlir::success();
}

/// Inspired by FuncOp::cloneInto
void cloneAttrsIntoZkirFunc(mlir::func::FuncOp src, zkir::FuncOp dest) {
  // Add the attributes of this function to dest (except visibility unless is
  // extern).
  llvm::MapVector<mlir::StringAttr, mlir::Attribute> newAttrMap;
  for (const auto &attr : dest->getAttrs()) {
    newAttrMap.insert({attr.getName(), attr.getValue()});
  }
  for (const auto &attr : src->getAttrs()) {
    /*if (attr.getName() == "sym_visibility")*/
    /*  continue;*/
    newAttrMap.insert({attr.getName(), attr.getValue()});
  }

  if (!newAttrMap.contains(mlir::StringAttr::get(src.getContext(), "extern")))
    newAttrMap.erase(mlir::StringAttr::get(src.getContext(), "sym_visibility"));

  auto newAttrs = llvm::to_vector(llvm::map_range(
      newAttrMap, [](std::pair<mlir::StringAttr, mlir::Attribute> attrPair) {
        return mlir::NamedAttribute(attrPair.first, attrPair.second);
      }));
  dest->setAttrs(mlir::DictionaryAttr::get(src.getContext(), newAttrs));
}

mlir::LogicalResult Zmir::FuncOpLowering::matchAndRewrite(
    mlir::func::FuncOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  auto type = op.getFunctionType();
  mlir::TypeConverter::SignatureConversion result(
      op.getFunctionType().getInputs().size());

  llvm::SmallVector<mlir::Type, 1> newResults;
  if (failed(
          getTypeConverter()->convertSignatureArgs(type.getInputs(), result)) ||
      failed(getTypeConverter()->convertTypes(type.getResults(), newResults)) ||
      failed(rewriter.convertRegionTypes(&op.getFunctionBody(),
                                         *getTypeConverter(), &result)))
    return mlir::failure();

  auto newType = mlir::FunctionType::get(
      rewriter.getContext(), result.getConvertedTypes(), newResults);

  auto newFuncOp =
      rewriter.replaceOpWithNewOp<zkir::FuncOp>(op, op.getNameAttr(), newType);
  cloneAttrsIntoZkirFunc(op, newFuncOp);
  newFuncOp.getRegion().takeBody(op.getRegion());
  return mlir::success();
}

mlir::LogicalResult Zmir::ReturnOpLowering::matchAndRewrite(
    mlir::func::ReturnOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<zkir::ReturnOp>(op, adaptor.getOperands());
  return mlir::success();
}

mlir::LogicalResult Zmir::CallOpLowering::matchAndRewrite(
    mlir::func::CallOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  llvm::SmallVector<mlir::Type> results;

  auto convRes = getTypeConverter()->convertTypes(
      op.getCalleeType().getResults(), results);
  if (mlir::failed(convRes))
    return op->emitError("failed to transform zmir types into zkir types");
  rewriter.replaceOpWithNewOp<zkir::CallOp>(op, adaptor.getCallee(), results,
                                            adaptor.getOperands());
  return mlir::success();
}

mlir::LogicalResult Zmir::CallIndirectOpLoweringInCompute::matchAndRewrite(
    mlir::func::CallIndirectOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto parent = op->getParentOfType<zkir::FuncOp>();
  if (!parent || parent.getName() != "compute")
    return mlir::failure(); // Don't operate on non compute calls

  auto callee = mlir::dyn_cast<Zmir::ConstructorRefOp>(
      adaptor.getCallee().getDefiningOp());
  if (!callee)
    return op->emitOpError()
           << "was expecting the callee comes from an zmir.constructor op";
  auto comp = callee.getComponentAttr();

  auto sym = mlir::SymbolRefAttr::get(
      comp.getAttr(), {mlir::SymbolRefAttr::get(parent.getNameAttr())});

  llvm::SmallVector<mlir::Type> types;
  auto convRes = getTypeConverter()->convertTypes(op.getResultTypes(), types);
  if (mlir::failed(convRes))
    return op->emitError("failed to transform zmir types into zkir types");
  mlir::ValueRange args(mlir::iterator_range(adaptor.getOperands().begin() + 1,
                                             adaptor.getOperands().end()));

  rewriter.replaceOpWithNewOp<zkir::CallOp>(op, sym, types, args);
  return mlir::success();
}

mlir::LogicalResult Zmir::CallIndirectOpLoweringInConstrain::matchAndRewrite(
    mlir::func::CallIndirectOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto parent = op->getParentOfType<zkir::FuncOp>();
  if (!parent || parent.getName() != "constrain")
    return mlir::failure(); // Don't operate on non compute calls

  if (!op->hasAttr("writes_into"))
    return mlir::failure();

  auto field = op->getAttrOfType<mlir::StringAttr>("writes_into");
  assert(field && "writes_into attribute exists but is not a symbol");

  auto callee = mlir::dyn_cast<Zmir::ConstructorRefOp>(
      adaptor.getCallee().getDefiningOp());
  assert(callee &&
         "was expecting the callee comes from an zmir.constructor op");
  auto constructorType = llvm::dyn_cast<mlir::FunctionType>(callee.getType());
  assert(constructorType && "was expecting a function type");
  assert(constructorType.getResults().size() == 1 &&
         "constructor type must have only one output");
  auto compType =
      llvm::dyn_cast<Zmir::ComponentType>(constructorType.getResults()[0]);

  auto fread = rewriter.create<zkir::FieldReadOp>(
      op.getLoc(), getTypeConverter()->convertType(compType),
      parent.getArgument(0), field);
  rewriter.replaceOp(op, fread);

  auto comp = callee.getComponentAttr();
  auto sym = mlir::SymbolRefAttr::get(
      comp.getAttr(), {mlir::SymbolRefAttr::get(parent.getNameAttr())});

  std::vector<mlir::Value> args;
  args.push_back(fread);
  args.insert(args.end(), adaptor.getOperands().begin() + 1,
              adaptor.getOperands().end());
  rewriter.create<zkir::CallOp>(op.getLoc(), sym, mlir::TypeRange(), args);

  return mlir::success();
}

mlir::LogicalResult Zmir::WriteFieldOpLowering::matchAndRewrite(
    Zmir::WriteFieldOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<zkir::FieldWriteOp>(
      op, adaptor.getComponent(), adaptor.getFieldNameAttr(), adaptor.getVal());
  return mlir::success();
}

mlir::LogicalResult Zmir::RemoveConstructorRefOp::matchAndRewrite(
    Zmir::ConstructorRefOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.eraseOp(op);
  return mlir::success();
}

mlir::LogicalResult Zmir::LowerReadFieldOp::matchAndRewrite(
    Zmir::ReadFieldOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<zkir::FieldReadOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getComponent(),
      adaptor.getFieldNameAttr());
  return mlir::success();
}

mlir::LogicalResult Zmir::LowerConstrainOp::matchAndRewrite(
    Zmir::ConstrainOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<zkir::EmitEqualityOp>(op, adaptor.getLhs(),
                                                    adaptor.getRhs());
  return mlir::success();
}

mlir::LogicalResult Zmir::LowerInRangeOp::matchAndRewrite(
    Zmir::InRangeOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto le = rewriter.create<zkir::CmpOp>(
      op.getLoc(),
      zkir::FeltCmpPredicateAttr::get(getContext(), zkir::FeltCmpPredicate::LE),
      adaptor.getLow(), adaptor.getMid());
  auto lt = rewriter.create<zkir::CmpOp>(
      op.getLoc(),
      zkir::FeltCmpPredicateAttr::get(getContext(), zkir::FeltCmpPredicate::LT),
      adaptor.getMid(), adaptor.getHigh());
  auto mul = rewriter.create<mlir::arith::MulIOp>(op.getLoc(), le, lt);
  auto conv = rewriter.create<zkir::FeltFromIntOp>(op.getLoc(), mul);
  rewriter.replaceOp(op, conv);

  return mlir::success();
}

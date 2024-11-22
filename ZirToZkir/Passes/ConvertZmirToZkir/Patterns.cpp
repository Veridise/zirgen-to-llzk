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

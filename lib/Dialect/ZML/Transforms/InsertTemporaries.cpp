
// Copyright 2024 Veridise, Inc.

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Dialect/ZML/BuiltIns/BuiltIns.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include "zklang/Dialect/ZML/IR/Types.h"
#include "zklang/Dialect/ZML/Transforms/PassDetail.h"
#include "zklang/Dialect/ZML/Typing/ZMIRTypeConverter.h"
#include <cassert>
#include <llvm/Support/Debug.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

namespace zkc::Zmir {

namespace {
class InsertTemporariesPass : public InsertTemporariesBase<InsertTemporariesPass> {

  mlir::FlatSymbolRefAttr
  createTempField(mlir::Location loc, mlir::Type type, mlir::OpBuilder &builder) {
    auto op = getOperation();
    mlir::SymbolTable st(op);
    auto desiredName = mlir::StringAttr::get(&getContext(), "$temp");
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(&op.getRegion().front().front());
    auto fieldDef = builder.create<FieldDefOp>(loc, desiredName, TypeAttr::get(type));
    return mlir::FlatSymbolRefAttr::get(&getContext(), st.insert(fieldDef));
  }

  void createTempFieldWrite(mlir::func::CallIndirectOp op, mlir::OpBuilder &builder) {
    // We are interested on the ones that we could not annotate
    if (op->hasAttr("writes_into")) {
      return;
    }
    auto results = op.getResultTypes();
    if (results.size() != 1) {
      return;
    }
    auto name = createTempField(op->getLoc(), results[0], builder);
    builder.setInsertionPointAfter(op);
    auto self = builder.create<GetSelfOp>(op.getLoc(), getOperation().getType());
    builder.create<WriteFieldOp>(op->getLoc(), self, name, op.getResult(0));

    op->setAttr("writes_into", name.getAttr());
  }

  template <typename Arr> void createTempFieldWrite(Arr op, mlir::OpBuilder &builder) {

    builder.setInsertionPointAfter(op);
    auto name = createTempField(op->getLoc(), op.getResult().getType(), builder);
    auto self = builder.create<GetSelfOp>(op.getLoc(), getOperation().getType());
    builder.create<WriteFieldOp>(op->getLoc(), self, name, op);
  }

  void runOnOperation() override {
    mlir::OpBuilder builder(&getContext());

    getOperation().walk([&](mlir::func::CallIndirectOp constructorCall) {
      createTempFieldWrite(constructorCall, builder);
    });

    getOperation().walk([&](Zmir::AllocArrayOp alloc) { createTempFieldWrite(alloc, builder); });

    getOperation().walk([&](Zmir::NewArrayOp newArr) { createTempFieldWrite(newArr, builder); });
  }
};

} // namespace

std::unique_ptr<OperationPass<ComponentOp>> createInsertTemporariesPass() {
  return std::make_unique<InsertTemporariesPass>();
}

} // namespace zkc::Zmir

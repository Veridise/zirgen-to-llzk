#include "BuiltIns.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace zkc::Zmir;

/// Holds information related to building
/// the built-ins
struct BuildContext {
  mlir::OpBuilder &builder;
  mlir::Location unk;

  explicit BuildContext(mlir::OpBuilder &builder)
      : builder(builder), unk(builder.getUnknownLoc()) {}

  /// Creates a component with no extra attributes and no parameters
  void createBasicComponent(mlir::StringRef name,
                            std::function<void(ComponentOp &)> buildComponent) {
    mlir::SmallVector<mlir::NamedAttribute> attributes;
    attributes.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(builder.getContext(), "builtin"),
        mlir::UnitAttr::get(builder.getContext())));

    ComponentOp op = builder.create<ComponentOp>(unk, name, IsBuiltIn{});
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);
    auto *block = builder.createBlock(&op.getRegion());
    builder.setInsertionPointToStart(block);

    buildComponent(op);
  }

  void fillOutBody(ComponentOp &op, mlir::ArrayRef<mlir::Type> argTypes,
                   mlir::ArrayRef<mlir::Type> results,
                   std::function<void(mlir::ValueRange)> buildBody) {

    auto funcType = builder.getFunctionType(argTypes, results);
    auto bodyOp = builder.create<mlir::func::FuncOp>(unk, "body", funcType);
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToStart(bodyOp.addEntryBlock());
    buildBody(bodyOp.getArguments());
  }
};

template <typename OpTy>
void addBinOp(BuildContext &ctx, mlir::StringRef name) {
  ctx.createBasicComponent(name, [&](ComponentOp &op) {
    auto val = ValType::get(ctx.builder.getContext());
    ctx.fillOutBody(op, {val, val}, {val}, [&](mlir::ValueRange args) {
      auto op = ctx.builder.create<OpTy>(ctx.unk, args[0], args[1]);
      ctx.builder.create<mlir::func::ReturnOp>(ctx.unk, mlir::ValueRange({op}));
    });
  });
}

template <typename OpTy>
void addUnaryOp(BuildContext &ctx, mlir::StringRef name) {
  ctx.createBasicComponent(name, [&](ComponentOp &op) {
    auto val = ValType::get(ctx.builder.getContext());
    ctx.fillOutBody(op, {val}, {val}, [&](mlir::ValueRange args) {
      auto op = ctx.builder.create<OpTy>(ctx.unk, args[0]);
      ctx.builder.create<mlir::func::ReturnOp>(ctx.unk, mlir::ValueRange({op}));
    });
  });
}

void addInRange(BuildContext &ctx) {
  ctx.createBasicComponent("InRange", [&](ComponentOp &op) {
    auto val = ValType::get(ctx.builder.getContext());
    ctx.fillOutBody(op, {val, val, val}, {val}, [&](mlir::ValueRange args) {
      auto op =
          ctx.builder.create<InRangeOp>(ctx.unk, args[0], args[1], args[2]);
      ctx.builder.create<mlir::func::ReturnOp>(ctx.unk, mlir::ValueRange({op}));
    });
  });
}

void addComponent(BuildContext &ctx) {
  ctx.createBasicComponent("Component", [&](ComponentOp &op) {
    auto componentType =
        ComponentType::get(ctx.builder.getContext(), op.getSymName());
    mlir::SmallVector<mlir::Type> args;
    ctx.fillOutBody(op, args, {componentType}, [&](mlir::ValueRange) {
      auto op = ctx.builder.create<GetSelfOp>(ctx.unk, componentType);
      ctx.builder.create<mlir::func::ReturnOp>(ctx.unk, mlir::ValueRange({op}));
    });
  });
}

void addNondetReg(BuildContext &ctx) {
  ctx.createBasicComponent("NondetReg", [&](ComponentOp &op) {
    auto val = ValType::get(ctx.builder.getContext());
    ctx.builder.create<FieldDefOp>(ctx.unk, "reg", val);
    ctx.fillOutBody(op, {val}, {val}, [&](mlir::ValueRange args) {
      ctx.builder.create<WriteFieldOp>(ctx.unk, "reg", args[0]);
      auto op = ctx.builder.create<ReadFieldOp>(ctx.unk, val, "reg");
      ctx.builder.create<mlir::func::ReturnOp>(ctx.unk, mlir::ValueRange({op}));
    });
  });
}

void addTrivial(BuildContext &ctx, mlir::StringRef name, mlir::Type type) {
  ctx.createBasicComponent(name, [&](ComponentOp &op) {
    ctx.fillOutBody(op, {type}, {type}, [&](mlir::ValueRange args) {
      ctx.builder.create<mlir::func::ReturnOp>(ctx.unk, args);
    });
  });
}

void zkc::Zmir::addBuiltins(mlir::OpBuilder &builder) {
  BuildContext ctx(builder);
  addInRange(ctx);
  addComponent(ctx);
  addNondetReg(ctx);

  addBinOp<BitAndOp>(ctx, "BitAnd");
  addBinOp<AddOp>(ctx, "Add");
  addBinOp<SubOp>(ctx, "Sub");
  addBinOp<MulOp>(ctx, "Mul");

  addUnaryOp<InvOp>(ctx, "Inv");
  addUnaryOp<IsZeroOp>(ctx, "Isz");
  addUnaryOp<NegOp>(ctx, "Neg");

  addTrivial(ctx, "Val", ValType::get(builder.getContext()));
  addTrivial(ctx, "String", StringType::get(builder.getContext()));
  /// The type 'Type' should also have a trivial constructor
  /// but how to handle this type at this level of abstraction
  /// is TBD.
}

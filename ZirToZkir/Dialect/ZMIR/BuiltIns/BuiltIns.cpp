#include "BuiltIns.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <unordered_set>

using namespace zkc::Zmir;

/// Holds information related to building
/// the built-ins
struct BuildContext {
  mlir::OpBuilder &builder;
  mlir::Location unk;

  explicit BuildContext(mlir::OpBuilder &builder)
      : builder(builder), unk(builder.getUnknownLoc()) {}

  /// Creates a component with no extra attributes and no parameters
  void
  createBasicComponent(mlir::StringRef name, std::function<void(ComponentOp &)> buildComponent) {
    ComponentOp op = builder.create<ComponentOp>(unk, name, IsBuiltIn{});
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);
    auto *block = builder.createBlock(&op.getRegion());
    builder.setInsertionPointToStart(block);

    buildComponent(op);
  }

  /// Creates a component with parameters
  void createParametricComponent(
      mlir::StringRef name, mlir::ArrayRef<mlir::StringRef> typeParams,
      mlir::ArrayRef<mlir::StringRef> constParams, std::function<void(ComponentOp &)> buildComponent
  ) {
    ComponentOp op = builder.create<ComponentOp>(unk, name, typeParams, constParams, IsBuiltIn{});
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);
    auto *block = builder.createBlock(&op.getRegion());
    builder.setInsertionPointToStart(block);

    buildComponent(op);
  }

  void fillOutBody(
      ComponentOp &op, mlir::ArrayRef<mlir::Type> argTypes, mlir::ArrayRef<mlir::Type> results,
      std::function<void(mlir::ValueRange)> buildBody
  ) {

    auto funcType = builder.getFunctionType(argTypes, results);
    std::vector<mlir::NamedAttribute> attrs = {mlir::NamedAttribute(
        builder.getStringAttr("sym_visibility"), builder.getStringAttr("nested")
    )};

    auto bodyOp = builder.create<mlir::func::FuncOp>(unk, op.getBodyFuncName(), funcType, attrs);
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToStart(bodyOp.addEntryBlock());
    buildBody(bodyOp.getArguments());
  }
};

template <typename OpTy> void addBinOp(BuildContext &ctx, mlir::StringRef name) {
  ctx.createBasicComponent(name, [&](ComponentOp &op) {
    auto componentType = ComponentType::get(ctx.builder.getContext(), op.getSymName());
    auto val = ValType::get(ctx.builder.getContext());
    // Special register where results are stored
    ctx.builder.create<FieldDefOp>(ctx.unk, "$super", val);
    ctx.fillOutBody(op, {val, val}, {val}, [&](mlir::ValueRange args) {
      // Reference to self
      auto self = ctx.builder.create<GetSelfOp>(ctx.unk, componentType);
      // Do the computation
      auto op = ctx.builder.create<OpTy>(ctx.unk, args[0], args[1]);
      // Store the result
      ctx.builder.create<WriteFieldOp>(ctx.unk, self, "$super", op);
      // Return self
      ctx.builder.create<mlir::func::ReturnOp>(ctx.unk, mlir::ValueRange({op}));
    });
  });
}

template <typename OpTy> void addUnaryOp(BuildContext &ctx, mlir::StringRef name) {
  ctx.createBasicComponent(name, [&](ComponentOp &op) {
    auto componentType = ComponentType::get(ctx.builder.getContext(), op.getSymName());
    auto val = ValType::get(ctx.builder.getContext());
    // Special register where results are stored
    ctx.builder.create<FieldDefOp>(ctx.unk, "$super", val);
    ctx.fillOutBody(op, {val}, {val}, [&](mlir::ValueRange args) {
      // Reference to self
      auto self = ctx.builder.create<GetSelfOp>(ctx.unk, componentType);
      // Do the computation
      auto op = ctx.builder.create<OpTy>(ctx.unk, args[0]);
      // Store the result
      ctx.builder.create<WriteFieldOp>(ctx.unk, self, "$super", op);
      // Return self
      ctx.builder.create<mlir::func::ReturnOp>(ctx.unk, mlir::ValueRange({op}));
    });
  });
}

void addInRange(BuildContext &ctx) {
  ctx.createBasicComponent("InRange", [&](ComponentOp &op) {
    auto componentType = ComponentType::get(ctx.builder.getContext(), op.getSymName());
    auto val = ValType::get(ctx.builder.getContext());
    // Special register where results are stored
    ctx.builder.create<FieldDefOp>(ctx.unk, "$super", val);
    ctx.fillOutBody(op, {val, val, val}, {componentType}, [&](mlir::ValueRange args) {
      // Reference to self
      auto self = ctx.builder.create<GetSelfOp>(ctx.unk, componentType);
      // Do the computation
      auto op = ctx.builder.create<InRangeOp>(ctx.unk, args[0], args[1], args[2]);
      // Store the result
      ctx.builder.create<WriteFieldOp>(ctx.unk, self, "$super", op);
      // Return self
      ctx.builder.create<mlir::func::ReturnOp>(ctx.unk, mlir::ValueRange({self}));
    });
  });
}

void addComponent(BuildContext &ctx) {
  ctx.createBasicComponent("Component", [&](ComponentOp &op) {
    auto componentType = ComponentType::get(ctx.builder.getContext(), op.getSymName());
    mlir::SmallVector<mlir::Type> args;
    ctx.fillOutBody(op, args, {componentType}, [&](mlir::ValueRange) {
      auto op = ctx.builder.create<GetSelfOp>(ctx.unk, componentType);
      ctx.builder.create<mlir::func::ReturnOp>(ctx.unk, mlir::ValueRange({op}));
    });
  });
}

void addNondetReg(BuildContext &ctx) {
  ctx.createBasicComponent("NondetReg", [&](ComponentOp &op) {
    auto componentType = ComponentType::get(ctx.builder.getContext(), op.getSymName());
    auto val = ValType::get(ctx.builder.getContext());
    ctx.builder.create<FieldDefOp>(ctx.unk, "reg", val);
    // Special register where results are stored
    ctx.builder.create<FieldDefOp>(ctx.unk, "$super", val);
    ctx.fillOutBody(op, {val}, {componentType}, [&](mlir::ValueRange args) {
      // Reference to self
      auto self = ctx.builder.create<GetSelfOp>(ctx.unk, componentType);
      ctx.builder.create<WriteFieldOp>(ctx.unk, self, "reg", args[0]);
      // Store the result
      ctx.builder.create<WriteFieldOp>(ctx.unk, self, "$super", args[0]);
      // Return self
      ctx.builder.create<mlir::func::ReturnOp>(ctx.unk, mlir::ValueRange({self}));
    });
  });
}

void addTrivial(BuildContext &ctx, mlir::StringRef name, mlir::Type type) {
  ctx.createBasicComponent(name, [&](ComponentOp &op) {
    auto componentType = ComponentType::get(ctx.builder.getContext(), op.getSymName());
    // Special register where results are stored
    ctx.builder.create<FieldDefOp>(ctx.unk, "$super", type);
    ctx.fillOutBody(op, {type}, {componentType}, [&](mlir::ValueRange args) {
      // Reference to self
      auto self = ctx.builder.create<GetSelfOp>(ctx.unk, componentType);
      // Store the result
      ctx.builder.create<WriteFieldOp>(ctx.unk, self, "$super", args[0]);
      ctx.builder.create<mlir::func::ReturnOp>(ctx.unk, mlir::ValueRange({self}));
    });
  });
}

void addArrayComponent(BuildContext &ctx) {

  auto symT = mlir::SymbolRefAttr::get(mlir::StringAttr::get(ctx.builder.getContext(), "T"));
  auto sizeVar = mlir::SymbolRefAttr::get(mlir::StringAttr::get(ctx.builder.getContext(), "N"));
  ctx.createParametricComponent("Array", {"T"}, {"N"}, [&](ComponentOp &op) {
    auto componentType =
        ComponentType::get(ctx.builder.getContext(), op.getSymName(), {symT}, {sizeVar});
    auto typeVar = TypeVarType::get(ctx.builder.getContext(), symT);
    auto type = BoundedArrayType::get(ctx.builder.getContext(), typeVar, sizeVar);
    // Special register where results are stored
    ctx.builder.create<FieldDefOp>(ctx.unk, "$super", type);

    ctx.fillOutBody(op, {type}, {componentType}, [&](mlir::ValueRange args) {
      // Reference to self
      auto self = ctx.builder.create<GetSelfOp>(ctx.unk, componentType);
      // Store the result
      ctx.builder.create<WriteFieldOp>(ctx.unk, self, "$super", args[0]);
      ctx.builder.create<mlir::func::ReturnOp>(ctx.unk, mlir::ValueRange({self}));
    });
  });
}

/// Adds the builtin operations that have not been overriden
/// checkOverride returns true if function has been overriden.
void zkc::Zmir::addBuiltins(
    mlir::OpBuilder &builder, std::function<bool(mlir::StringRef)> checkOverride
) {
  BuildContext ctx(builder);
  if (!checkOverride("InRange")) {
    addInRange(ctx);
  }
  if (!checkOverride("Component")) {
    addComponent(ctx);
  }
  if (!checkOverride("NondetReg")) {
    addNondetReg(ctx);
  }

  if (!checkOverride("BitAnd")) {
    addBinOp<BitAndOp>(ctx, "BitAnd");
  }
  if (!checkOverride("Add")) {
    addBinOp<AddOp>(ctx, "Add");
  }
  if (!checkOverride("Sub")) {
    addBinOp<SubOp>(ctx, "Sub");
  }
  if (!checkOverride("Mul")) {
    addBinOp<MulOp>(ctx, "Mul");
  }

  if (!checkOverride("Inv")) {
    addUnaryOp<InvOp>(ctx, "Inv");
  }
  if (!checkOverride("Isz")) {
    addUnaryOp<IsZeroOp>(ctx, "Isz");
  }
  if (!checkOverride("Neg")) {
    addUnaryOp<NegOp>(ctx, "Neg");
  }

  addTrivial(ctx, "Val", ValType::get(builder.getContext()));
  // addTrivial(ctx, "String", StringType::get(builder.getContext()));
  /// The type 'Type' should also have a trivial constructor
  /// but how to handle this type at this level of abstraction
  /// is TBD.
  // addArrayComponent(ctx);
}

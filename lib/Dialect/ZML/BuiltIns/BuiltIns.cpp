#include "zklang/Dialect/ZML/BuiltIns/BuiltIns.h"
#include "zklang/Dialect/ZHL/Typing/TypeBindings.h"
#include "zklang/Dialect/ZML/IR/Builder.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include "zklang/Dialect/ZML/IR/Types.h"
#include <functional>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ValueRange.h>
#include <unordered_set>

using namespace zkc::Zmir;

ComponentBuilder &builtinCommon(ComponentBuilder &builder) { return builder.isBuiltin(); }

ComponentBuilder &selfConstructs(ComponentBuilder &builder, mlir::Type type) {
  return builder.fillBody({type}, {type}, [&](mlir::ValueRange args, mlir::OpBuilder &builder) {
    // Reference to self
    auto self = builder.create<GetSelfOp>(builder.getUnknownLoc(), type);
    // Construct Component superType
    mlir::FunctionType constructor =
        builder.getFunctionType({}, ComponentType::Component(builder.getContext()));
    auto ref = builder.create<ConstructorRefOp>(
        builder.getUnknownLoc(), constructor,
        mlir::SymbolRefAttr::get(builder.getStringAttr("Component")), builder.getUnitAttr()
    );
    auto comp = builder.create<mlir::func::CallIndirectOp>(builder.getUnknownLoc(), ref);
    // Store the result
    builder.create<WriteFieldOp>(builder.getUnknownLoc(), self, "$super", comp.getResult(0));
    auto super = builder.create<ReadFieldOp>(
        builder.getUnknownLoc(), comp.getResultTypes().front(), self, "$super"
    );
    builder.create<ConstrainCallOp>(builder.getUnknownLoc(), super, mlir::ValueRange());
    // Return self
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({args[0]}));
  });
}

template <typename OpTy> void addBinOp(mlir::OpBuilder &builder, mlir::StringRef name) {
  auto superType = ComponentType::Val(builder.getContext());
  auto componentType = ComponentType::get(builder.getContext(), name, superType);

  builtinCommon(ComponentBuilder()
                    .name(name)
                    .field("$super", superType)
                    .fillBody(
                        {superType, superType}, {componentType},
                        [&](mlir::ValueRange args, mlir::OpBuilder &builder) {
    // Reference to self
    auto self = builder.create<GetSelfOp>(builder.getUnknownLoc(), componentType);
    // Do the computation
    auto op = builder.create<OpTy>(builder.getUnknownLoc(), superType, args[0], args[1]);
    // Store the result
    builder.create<WriteFieldOp>(builder.getUnknownLoc(), self, "$super", op);
    // Return self
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({self}));
  }
                    )
  ).build(builder);
}

template <typename OpTy> void addUnaryOp(mlir::OpBuilder &builder, mlir::StringRef name) {
  auto superType = ComponentType::Val(builder.getContext());
  auto componentType = ComponentType::get(builder.getContext(), name, superType);

  builtinCommon(ComponentBuilder()
                    .name(name)
                    .field("$super", superType)
                    .fillBody(
                        {superType}, {componentType},
                        [&](mlir::ValueRange args, mlir::OpBuilder &builder) {
    // Reference to self
    auto self = builder.create<GetSelfOp>(builder.getUnknownLoc(), componentType);
    // Do the computation
    auto op = builder.create<OpTy>(builder.getUnknownLoc(), superType, args[0]);
    // Store the result
    builder.create<WriteFieldOp>(builder.getUnknownLoc(), self, "$super", op);
    // Return self
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({self}));
  }
                    )
  ).build(builder);
}

void addInRange(mlir::OpBuilder &builder) {
  auto superType = ComponentType::Val(builder.getContext());
  auto componentType = ComponentType::get(builder.getContext(), "InRange", superType);

  builtinCommon(ComponentBuilder()
                    .name("InRange")
                    .field("$super", superType)
                    .fillBody(
                        {superType, superType, superType}, {componentType},
                        [&](mlir::ValueRange args, mlir::OpBuilder &builder) {
    // Reference to self
    auto self = builder.create<GetSelfOp>(builder.getUnknownLoc(), componentType);
    // Do the computation
    auto op =
        builder.create<InRangeOp>(builder.getUnknownLoc(), superType, args[0], args[1], args[2]);
    // Store the result
    builder.create<WriteFieldOp>(builder.getUnknownLoc(), self, "$super", op);
    // Return self
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({self}));
  }
                    )
  ).build(builder);
}

void addComponent(mlir::OpBuilder &builder) {
  auto componentType = ComponentType::Component(builder.getContext());
  mlir::SmallVector<mlir::Type> args;

  builtinCommon(ComponentBuilder()
                    .name("Component")
                    .fillBody(
                        args, {componentType},
                        [&](mlir::ValueRange args, mlir::OpBuilder &builder) {
    auto op = builder.create<GetSelfOp>(builder.getUnknownLoc(), componentType);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({op}));
  }
                    )
  ).build(builder);
}

void addNondetReg(mlir::OpBuilder &builder) {
  auto superType = ComponentType::Val(builder.getContext());
  auto componentType = ComponentType::get(builder.getContext(), "NondetReg", superType);

  builtinCommon(ComponentBuilder()
                    .name("NondetReg")
                    .field("$super", superType)
                    .field("reg", superType)
                    .fillBody(
                        {superType}, {componentType},
                        [&](mlir::ValueRange args, mlir::OpBuilder &builder) {
    // Reference to self
    auto self = builder.create<GetSelfOp>(builder.getUnknownLoc(), componentType);
    builder.create<WriteFieldOp>(builder.getUnknownLoc(), self, "reg", args[0]);
    // Store the result
    builder.create<WriteFieldOp>(builder.getUnknownLoc(), self, "$super", args[0]);
    // Return self
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({self}));
  }
                    )
  ).build(builder);
}

void addTrivial(mlir::OpBuilder &builder, mlir::StringRef name) {
  auto superType = ComponentType::Component(builder.getContext());
  auto componentType = ComponentType::get(builder.getContext(), name, superType);

  selfConstructs(
      builtinCommon(ComponentBuilder().name(name).field("$super", superType)

      ),
      componentType
  )
      .build(builder);
}

void addArrayComponent(mlir::OpBuilder &builder) {
  auto componentType = ComponentType::Array(builder.getContext());
  auto superType = ComponentType::Component(builder.getContext());

  selfConstructs(
      builtinCommon(
          ComponentBuilder().name("Array").forceGeneric().typeParam("T").typeParam("N").field(
              "$super", superType
          )
      ),
      componentType
  )
      .build(builder);
}
#define MAYBE(name) if (definedNames.find(name) == definedNames.end())

void zkc::Zmir::addBuiltinBindings(
    zhl::TypeBindings &bindings, const std::unordered_set<std::string_view> &definedNames
) {
  auto &Val = bindings.Create("Val", bindings.Component());
  const_cast<zhl::TypeBinding &>(Val).selfConstructs();
  MAYBE("String") {
    auto &String = bindings.Create("String", bindings.Component());
    const_cast<zhl::TypeBinding &>(String).selfConstructs();
  }
  auto &Type = bindings.Create("Type", bindings.Component());

  MAYBE("NondetReg")
  bindings.Create(
      "NondetReg", Val, zhl::ParamsMap(), zhl::ParamsMap({{{"v", 0}, Val}}), zhl::MembersMap()
  );
  MAYBE("InRange")
  bindings.Create(
      "InRange", Val, zhl::ParamsMap(),
      zhl::ParamsMap({{{"low", 0}, Val}, {{"mid", 1}, Val}, {{"high", 2}, Val}}), zhl::MembersMap()
  );
  MAYBE("BitAnd")
  bindings.Create(
      "BitAnd", Val, zhl::ParamsMap(), zhl::ParamsMap({{{"lhs", 0}, Val}, {{"rhs", 1}, Val}}),
      zhl::MembersMap()
  );
  MAYBE("Add")
  bindings.Create(
      "Add", Val, zhl::ParamsMap(), zhl::ParamsMap({{{"lhs", 0}, Val}, {{"rhs", 1}, Val}}),
      zhl::MembersMap()
  );
  MAYBE("Sub")
  bindings.Create(
      "Sub", Val, zhl::ParamsMap(), zhl::ParamsMap({{{"lhs", 0}, Val}, {{"rhs", 1}, Val}}),
      zhl::MembersMap()
  );
  MAYBE("Mul")
  bindings.Create(
      "Mul", Val, zhl::ParamsMap(), zhl::ParamsMap({{{"lhs", 0}, Val}, {{"rhs", 1}, Val}}),
      zhl::MembersMap()
  );
  MAYBE("Inv")
  bindings.Create(
      "Inv", Val, zhl::ParamsMap(), zhl::ParamsMap({{{"v", 0}, Val}}), zhl::MembersMap()
  );
  MAYBE("Isz")
  bindings.Create(
      "Isz", Val, zhl::ParamsMap(), zhl::ParamsMap({{{"v", 0}, Val}}), zhl::MembersMap()
  );
  MAYBE("Neg")
  bindings.Create(
      "Neg", Val, zhl::ParamsMap(), zhl::ParamsMap({{{"v", 0}, Val}}), zhl::MembersMap()
  );
  MAYBE("Array") {
    auto &Array = bindings.Create(
        "Array", bindings.Component(), zhl::ParamsMap({{{"T", 0}, Type}, {{"N", 1}, Val}})
    );
    const_cast<zhl::TypeBinding &>(Array).selfConstructs();
  }
}

/// Adds the builtin operations that have not been overriden
void zkc::Zmir::addBuiltins(
    mlir::OpBuilder &builder, const std::unordered_set<std::string_view> &definedNames
) {
  assert(definedNames.find("Component") == definedNames.end() && "Can't redefine Component type");
  addComponent(builder);

  assert(definedNames.find("Val") == definedNames.end() && "Can't redefine Val type");
  addTrivial(builder, "Val");
  MAYBE("String") { addTrivial(builder, "String"); }

  MAYBE("NondetReg") { addNondetReg(builder); }
  MAYBE("InRange") { addInRange(builder); }
  MAYBE("BitAnd") { addBinOp<BitAndOp>(builder, "BitAnd"); }
  MAYBE("Add") { addBinOp<AddOp>(builder, "Add"); }
  MAYBE("Sub") { addBinOp<SubOp>(builder, "Sub"); }
  MAYBE("Mul") { addBinOp<MulOp>(builder, "Mul"); }
  MAYBE("Inv") { addUnaryOp<InvOp>(builder, "Inv"); }
  MAYBE("Isz") { addUnaryOp<IsZeroOp>(builder, "Isz"); }
  MAYBE("Neg") { addUnaryOp<NegOp>(builder, "Neg"); }
  MAYBE("Array") { addArrayComponent(builder); }
}

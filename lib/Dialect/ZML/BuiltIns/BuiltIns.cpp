#include <functional>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ValueRange.h>
#include <unordered_set>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>
#include <zklang/Dialect/ZML/BuiltIns/BuiltIns.h>
#include <zklang/Dialect/ZML/IR/Builder.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/Types.h>

using namespace zml;

ComponentBuilder &builtinCommon(ComponentBuilder &builder) { return builder.isBuiltin(); }

ComponentBuilder &selfConstructs(ComponentBuilder &builder, mlir::Type type) {
  return builder.fillBody(
      {type}, {type},
      [type = type](mlir::ValueRange args, mlir::OpBuilder &builder) {
    // Reference to self
    auto self = builder.create<SelfOp>(builder.getUnknownLoc(), type);
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
  }
  );
}

template <typename OpTy>
void addBinOpCommon(mlir::OpBuilder &builder, mlir::StringRef name, ComponentType superType) {
  auto componentType = ComponentType::get(builder.getContext(), name, superType, true);

  builtinCommon(ComponentBuilder()
                    .name(name)
                    .field("$super", superType)
                    .fillBody(
                        {superType, superType}, {componentType},
                        [&](mlir::ValueRange args, mlir::OpBuilder &builder) {
    // Reference to self
    auto self = builder.create<SelfOp>(builder.getUnknownLoc(), componentType);
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
template <typename OpTy> void addBinOp(mlir::OpBuilder &builder, mlir::StringRef name) {
  auto superType = ComponentType::Val(builder.getContext());
  addBinOpCommon<OpTy>(builder, name, superType);
}

template <typename OpTy> void addExtBinOp(mlir::OpBuilder &builder, mlir::StringRef name) {
  auto superType = ComponentType::ExtVal(builder.getContext());
  addBinOpCommon<OpTy>(builder, name, superType);
}

template <typename OpTy>
void addUnaryOpCommon(mlir::OpBuilder &builder, mlir::StringRef name, ComponentType superType) {
  auto componentType = ComponentType::get(builder.getContext(), name, superType, true);

  builtinCommon(ComponentBuilder()
                    .name(name)
                    .field("$super", superType)
                    .fillBody(
                        {superType}, {componentType},
                        [&](mlir::ValueRange args, mlir::OpBuilder &builder) {
    // Reference to self
    auto self = builder.create<SelfOp>(builder.getUnknownLoc(), componentType);
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

template <typename OpTy> void addUnaryOp(mlir::OpBuilder &builder, mlir::StringRef name) {
  auto superType = ComponentType::Val(builder.getContext());
  addUnaryOpCommon<OpTy>(builder, name, superType);
}

template <typename OpTy> void addExtUnaryOp(mlir::OpBuilder &builder, mlir::StringRef name) {
  auto superType = ComponentType::ExtVal(builder.getContext());
  addUnaryOpCommon<OpTy>(builder, name, superType);
}

void addInRange(mlir::OpBuilder &builder) {
  auto superType = ComponentType::Val(builder.getContext());
  auto componentType = ComponentType::get(builder.getContext(), "InRange", superType, true);

  builtinCommon(ComponentBuilder()
                    .name("InRange")
                    .field("$super", superType)
                    .fillBody(
                        {superType, superType, superType}, {componentType},
                        [&](mlir::ValueRange args, mlir::OpBuilder &builder) {
    // Reference to self
    auto self = builder.create<SelfOp>(builder.getUnknownLoc(), componentType);
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
                        [&](mlir::ValueRange args, mlir::OpBuilder &b) {
    auto op = builder.create<SelfOp>(builder.getUnknownLoc(), componentType);
    b.create<mlir::func::ReturnOp>(b.getUnknownLoc(), mlir::ValueRange({op}));
  }
                    )
  ).build(builder);
}

void addNondetRegCommon(mlir::OpBuilder &builder, mlir::StringRef name, ComponentType superType) {
  auto componentType = ComponentType::get(builder.getContext(), name, superType, true);

  builtinCommon(ComponentBuilder()
                    .name(name)
                    .field("$super", superType)
                    .field("reg", superType)
                    .fillBody(
                        {superType}, {componentType},
                        [&](mlir::ValueRange args, mlir::OpBuilder &b) {
    // Reference to self
    auto self = builder.create<SelfOp>(builder.getUnknownLoc(), componentType);
    b.create<WriteFieldOp>(b.getUnknownLoc(), self, "reg", args[0]);
    // Store the result
    b.create<WriteFieldOp>(b.getUnknownLoc(), self, "$super", args[0]);
    // Return self
    b.create<mlir::func::ReturnOp>(b.getUnknownLoc(), mlir::ValueRange({self}));
  }
                    )
  ).build(builder);
}

void addNondetReg(mlir::OpBuilder &builder) {
  auto superType = ComponentType::Val(builder.getContext());
  addNondetRegCommon(builder, "NondetReg", superType);
}

void addNondetExtReg(mlir::OpBuilder &builder) {
  auto superType = ComponentType::ExtVal(builder.getContext());
  addNondetRegCommon(builder, "NondetExtReg", superType);
}

void addMakeExt(mlir::OpBuilder &builder) {
  auto superType = ComponentType::ExtVal(builder.getContext());
  auto valType = ComponentType::Val(builder.getContext());
  auto componentType = ComponentType::get(builder.getContext(), "MakeExt", superType, true);

  builtinCommon(ComponentBuilder()
                    .name("MakeExt")
                    .field("$super", superType)
                    .fillBody(
                        {valType}, {componentType},
                        [&](mlir::ValueRange args, mlir::OpBuilder &) {
    // Reference to self
    auto self = builder.create<SelfOp>(builder.getUnknownLoc(), componentType);
    auto ext = builder.create<MakeExtOp>(builder.getUnknownLoc(), superType, args[0]);
    // Store the result
    builder.create<WriteFieldOp>(builder.getUnknownLoc(), self, "$super", ext);
    // Return self
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({self}));
  }
                    )
  ).build(builder);
}

void addEqzExt(mlir::OpBuilder &builder) {
  auto superType = ComponentType::Component(builder.getContext());
  auto valType = ComponentType::ExtVal(builder.getContext());

  auto componentType = ComponentType::get(builder.getContext(), "EqzExt", superType, true);

  builtinCommon(ComponentBuilder()
                    .name("EqzExt")
                    .field("$super", superType)
                    .fillBody(
                        {valType}, {componentType},
                        [&](mlir::ValueRange args, mlir::OpBuilder &) {
    // Reference to self
    auto self = builder.create<SelfOp>(builder.getUnknownLoc(), componentType);
    builder.create<EqzExtOp>(builder.getUnknownLoc(), args[0]);
    auto compCtorRef = builder.create<ConstructorRefOp>(
        builder.getUnknownLoc(), superType.getName(),
        builder.getFunctionType(mlir::TypeRange(), superType), true
    );
    auto comp = builder.create<mlir::func::CallIndirectOp>(
        builder.getUnknownLoc(), compCtorRef, mlir::ValueRange()
    );
    // Store the result
    builder.create<WriteFieldOp>(builder.getUnknownLoc(), self, "$super", comp.getResult(0));
    // Return self
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({self}));
  }
                    )
  ).build(builder);
}

void addTrivial(mlir::OpBuilder &builder, mlir::StringRef name) {
  auto superType = ComponentType::Component(builder.getContext());
  auto componentType = ComponentType::get(builder.getContext(), name, superType, true);

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

void zml::addBuiltinBindings(
    zhl::TypeBindings &bindings, const std::unordered_set<std::string_view> &definedNames
) {
  auto &Val = bindings.CreateBuiltin("Val", bindings.Component());
  const_cast<zhl::TypeBinding &>(Val).selfConstructs();
  auto &ExtVal = bindings.CreateBuiltin("ExtVal", bindings.Component());
  const_cast<zhl::TypeBinding &>(ExtVal).selfConstructs();
  MAYBE("String") {
    auto &String = bindings.CreateBuiltin("String", bindings.Component());
    const_cast<zhl::TypeBinding &>(String).selfConstructs();
  }
  auto &Type = bindings.CreateBuiltin("Type", bindings.Component());
  auto T = zhl::TypeBinding::MakeGenericParam(Type, "T");
  auto N = zhl::TypeBinding::MakeGenericParam(Val, "N");

  MAYBE("NondetReg") {
    bindings.CreateBuiltin(
        "NondetReg", Val, zhl::ParamsMap(), zhl::ParamsMap({{"v", {Val, 0}}}), zhl::MembersMap()
    );
  }
  MAYBE("NondetExtReg") {
    bindings.CreateBuiltin(
        "NondetExtReg", ExtVal, zhl::ParamsMap(), zhl::ParamsMap({{"v", {ExtVal, 0}}}),
        zhl::MembersMap()
    );
  }
  MAYBE("InRange") {
    bindings.CreateBuiltin(
        "InRange", Val, zhl::ParamsMap(),
        zhl::ParamsMap({{"low", {Val, 0}}, {"mid", {Val, 1}}, {"high", {Val, 2}}}),
        zhl::MembersMap()
    );
  }
  MAYBE("BitAnd") {
    bindings.CreateBuiltin(
        "BitAnd", Val, zhl::ParamsMap(), zhl::ParamsMap({{"lhs", {Val, 0}}, {"rhs", {Val, 1}}}),
        zhl::MembersMap()
    );
  }
  MAYBE("Add") {
    bindings.CreateBuiltin(
        "Add", Val, zhl::ParamsMap(), zhl::ParamsMap({{"lhs", {Val, 0}}, {"rhs", {Val, 1}}}),
        zhl::MembersMap()
    );
  }
  MAYBE("ExtAdd") {
    bindings.CreateBuiltin(
        "ExtAdd", ExtVal, zhl::ParamsMap(),
        zhl::ParamsMap({{"lhs", {ExtVal, 0}}, {"rhs", {ExtVal, 1}}}), zhl::MembersMap()
    );
  }
  MAYBE("Sub") {
    bindings.CreateBuiltin(
        "Sub", Val, zhl::ParamsMap(), zhl::ParamsMap({{"lhs", {Val, 0}}, {"rhs", {Val, 1}}}),
        zhl::MembersMap()
    );
  }
  MAYBE("ExtSub") {
    bindings.CreateBuiltin(
        "ExtSub", ExtVal, zhl::ParamsMap(),
        zhl::ParamsMap({{"lhs", {ExtVal, 0}}, {"rhs", {ExtVal, 1}}}), zhl::MembersMap()
    );
  }
  MAYBE("Mul") {
    bindings.CreateBuiltin(
        "Mul", Val, zhl::ParamsMap(), zhl::ParamsMap({{"lhs", {Val, 0}}, {"rhs", {Val, 1}}}),
        zhl::MembersMap()
    );
  }
  MAYBE("ExtMul") {
    bindings.CreateBuiltin(
        "ExtMul", ExtVal, zhl::ParamsMap(),
        zhl::ParamsMap({{"lhs", {ExtVal, 0}}, {"rhs", {ExtVal, 1}}}), zhl::MembersMap()
    );
  }
  MAYBE("Mod") {
    bindings.CreateBuiltin(
        "Mod", Val, zhl::ParamsMap(), zhl::ParamsMap({{"lhs", {Val, 0}}, {"rhs", {Val, 1}}}),
        zhl::MembersMap()
    );
  }
  MAYBE("Inv") {
    bindings.CreateBuiltin(
        "Inv", Val, zhl::ParamsMap(), zhl::ParamsMap({{"v", {Val, 0}}}), zhl::MembersMap()
    );
  }
  MAYBE("ExtInv") {
    bindings.CreateBuiltin(
        "ExtInv", ExtVal, zhl::ParamsMap(), zhl::ParamsMap({{"v", {ExtVal, 0}}}), zhl::MembersMap()
    );
  }
  MAYBE("Isz") {
    bindings.CreateBuiltin(
        "Isz", Val, zhl::ParamsMap(), zhl::ParamsMap({{"v", {Val, 0}}}), zhl::MembersMap()
    );
  }
  MAYBE("Neg") {
    bindings.CreateBuiltin(
        "Neg", Val, zhl::ParamsMap(), zhl::ParamsMap({{"v", {Val, 0}}}), zhl::MembersMap()
    );
  }
  MAYBE("MakeExt") {
    bindings.CreateBuiltin(
        "MakeExt", ExtVal, zhl::ParamsMap(), zhl::ParamsMap({{"v", {Val, 0}}}), zhl::MembersMap()
    );
  }
  MAYBE("EqzExt") {
    bindings.CreateBuiltin(
        "EqzExt", bindings.Component(), zhl::ParamsMap(), zhl::ParamsMap({{"v", {ExtVal, 0}}}),
        zhl::MembersMap()
    );
  }
  MAYBE("Array") {
    auto &Array = bindings.CreateBuiltin(
        "Array", bindings.Component(), zhl::ParamsMap({{"T", {T, 0}}, {"N", {N, 1}}})
    );
    const_cast<zhl::TypeBinding &>(Array).selfConstructs();
  }
}

/// Adds the builtin operations that have not been overriden
void zml::addBuiltins(
    mlir::OpBuilder &builder, const std::unordered_set<std::string_view> &definedNames
) {
  assert(definedNames.find("Component") == definedNames.end() && "Can't redefine Component type");
  addComponent(builder);

  assert(definedNames.find("Val") == definedNames.end() && "Can't redefine Val type");
  assert(definedNames.find("ExtVal") == definedNames.end() && "Can't redefine ExtVal type");
  addTrivial(builder, "Val");
  addTrivial(builder, "ExtVal");
  MAYBE("String") { addTrivial(builder, "String"); }

  MAYBE("NondetReg") { addNondetReg(builder); }
  MAYBE("NondetExtReg") { addNondetExtReg(builder); }
  MAYBE("MakeExt") { addMakeExt(builder); }
  MAYBE("EqzExt") { addEqzExt(builder); }
  MAYBE("InRange") { addInRange(builder); }
  MAYBE("BitAnd") { addBinOp<BitAndOp>(builder, "BitAnd"); }
  MAYBE("Add") { addBinOp<AddOp>(builder, "Add"); }
  MAYBE("Sub") { addBinOp<SubOp>(builder, "Sub"); }
  MAYBE("Mul") { addBinOp<MulOp>(builder, "Mul"); }
  MAYBE("ExtAdd") { addExtBinOp<ExtAddOp>(builder, "ExtAdd"); }
  MAYBE("ExtSub") { addExtBinOp<ExtSubOp>(builder, "ExtSub"); }
  MAYBE("ExtMul") { addExtBinOp<ExtMulOp>(builder, "ExtMul"); }
  MAYBE("Mod") { addBinOp<ModOp>(builder, "Mod"); }
  MAYBE("Inv") { addUnaryOp<InvOp>(builder, "Inv"); }
  MAYBE("ExtInv") { addExtUnaryOp<ExtInvOp>(builder, "ExtInv"); }
  MAYBE("Isz") { addUnaryOp<IsZeroOp>(builder, "Isz"); }
  MAYBE("Neg") { addUnaryOp<NegOp>(builder, "Neg"); }
  MAYBE("Array") { addArrayComponent(builder); }
}

//===- BuiltIns.cpp - Builtins definitions ----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Felt/IR/Types.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Polymorphic/IR/Types.h>
#include <llzk/Dialect/String/IR/Types.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Types.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <unordered_set>
#include <zklang/Dialect/LLZK/Builder.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>
#include <zklang/Dialect/ZML/BuiltIns/BuiltIns.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/TypeInterfaces.h>
#include <zklang/Dialect/ZML/IR/Types.h>

using namespace zml;
using namespace llzk;
using namespace zml::builtins;

ComponentLike zml::builtins::Val(mlir::MLIRContext *ctx) {
  return mlir::cast<ComponentLike>(llzk::felt::FeltType::get(ctx));
}

ComponentLike zml::builtins::String(mlir::MLIRContext *ctx) {
  return mlir::cast<ComponentLike>(llzk::string::StringType::get(ctx));
}

ComponentLike zml::builtins::ExtVal(mlir::MLIRContext *ctx) { return ExtValType::get(ctx); }

ComponentLike zml::builtins::Component(mlir::MLIRContext *ctx) { return RootType::get(ctx); }

ComponentLike
zml::builtins::Array(mlir::Type inner, mlir::Attribute size, const mlir::TypeConverter &tc) {
  // If the inner type is another array we need to combine them
  if (auto innerArr = mlir::dyn_cast<llzk::array::ArrayType>(inner)) {
    auto actualInner = innerArr.getElementType();
    llvm::SmallVector<mlir::Attribute> dims(innerArr.getDimensionSizes());
    dims.insert(dims.begin(), size);
    assert(dims.size() == innerArr.getDimensionSizes().size() + 1);
    return mlir::cast<ComponentLike>(llzk::array::ArrayType::get(actualInner, dims));
  }

  mlir::Type dest = llzk::isValidArrayElemType(inner) ? inner : tc.convertType(inner);
  return mlir::cast<ComponentLike>(llzk::array::ArrayType::get(dest, {size}));
}

static mlir::Location getLocationForBuiltin(mlir::Builder &builder, mlir::StringRef name) {
  mlir::Twine filename = "<builtin " + name + ">";
  return mlir::Location(
      mlir::FileLineColLoc::get(builder.getContext(), builder.getStringAttr(filename), 0, 0)
  );
}

ComponentBuilder &builtinCommon(ComponentBuilder &builder) { return builder.isBuiltin(); }

ComponentBuilder &selfConstructs(
    ComponentBuilder &builder, mlir::Type type, mlir::Type structType, mlir::StringRef name
) {
  mlir::Builder mlirBuilder(type.getContext());
  return builder.location(getLocationForBuiltin(mlirBuilder, name))
      .fillBody(
          {type}, {structType},
          [structType,
           name](mlir::ValueRange args, mlir::OpBuilder &bldr, const mlir::TypeConverter &) {
    mlir::Location loc = getLocationForBuiltin(bldr, name);
    // Reference to self
    auto self = bldr.create<component::CreateStructOp>(loc, structType);
    bldr.create<component::FieldWriteOp>(loc, self, "$super", args[0]);
    // Return self
    bldr.create<function::ReturnOp>(loc, mlir::ValueRange({self}));
  }
      );
}

template <typename OpTy>
void addBinOpCommon(
    mlir::OpBuilder &builder, const mlir::TypeConverter &TC, mlir::StringRef name,
    ComponentLike superType
) {
  auto componentType = ComplexComponentType::get(builder.getContext(), name, superType, true);

  mlir::Location loc = getLocationForBuiltin(builder, name);
  builtinCommon(
      ComponentBuilder()
          .name(name)
          .field("$super", superType)
          .location(loc)
          .fillBody(
              {superType, superType}, {componentType},
              [&superType, &componentType,
               &loc](mlir::ValueRange args, mlir::OpBuilder &bldr, const mlir::TypeConverter &) {
    // Reference to self
    auto self = bldr.create<SelfOp>(loc, componentType);
    // Do the computation
    auto op = bldr.create<OpTy>(loc, superType, args[0], args[1]);
    // Store the result
    bldr.create<WriteFieldOp>(loc, self, "$super", op);
    // Return self
    bldr.create<mlir::func::ReturnOp>(loc, mlir::ValueRange({self}));
  }
          )
  ).build(builder, TC);
}
template <typename OpTy>
void addBinOp(mlir::OpBuilder &builder, const mlir::TypeConverter &TC, mlir::StringRef name) {
  addBinOpCommon<OpTy>(builder, TC, name, Val(builder));
}

template <typename OpTy>
void addExtBinOp(mlir::OpBuilder &builder, const mlir::TypeConverter &TC, mlir::StringRef name) {
  addBinOpCommon<OpTy>(builder, TC, name, ExtVal(builder));
}

template <typename OpTy>
void addUnaryOpCommon(
    mlir::OpBuilder &builder, const mlir::TypeConverter &TC, mlir::StringRef name,
    ComponentLike superType
) {
  auto componentType = ComplexComponentType::get(builder.getContext(), name, superType, true);

  mlir::Location loc = getLocationForBuiltin(builder, name);
  builtinCommon(
      ComponentBuilder()
          .name(name)
          .field("$super", superType)
          .location(loc)
          .fillBody(
              {superType}, {componentType},
              [&superType, &componentType,
               &loc](mlir::ValueRange args, mlir::OpBuilder &bldr, const mlir::TypeConverter &) {
    // Reference to self
    auto self = bldr.create<SelfOp>(loc, componentType);
    // Do the computation
    auto op = bldr.create<OpTy>(loc, superType, args[0]);
    // Store the result
    bldr.create<WriteFieldOp>(loc, self, "$super", op);
    // Return self
    bldr.create<mlir::func::ReturnOp>(loc, mlir::ValueRange({self}));
  }
          )
  ).build(builder, TC);
}

template <typename OpTy>
void addUnaryOp(mlir::OpBuilder &builder, const mlir::TypeConverter &TC, mlir::StringRef name) {
  addUnaryOpCommon<OpTy>(builder, TC, name, Val(builder));
}

template <typename OpTy>
void addExtUnaryOp(mlir::OpBuilder &builder, const mlir::TypeConverter &TC, mlir::StringRef name) {
  addUnaryOpCommon<OpTy>(builder, TC, name, ExtVal(builder));
}

void addInRange(mlir::OpBuilder &builder, const mlir::TypeConverter &TC) {
  auto superType = Val(builder);
  auto componentType = ComplexComponentType::get(builder.getContext(), "InRange", superType, true);
  mlir::Location loc = getLocationForBuiltin(builder, componentType.getName().getValue());
  builtinCommon(
      ComponentBuilder()
          .name(componentType.getName().getValue())
          .location(loc)
          .field("$super", superType)
          .fillBody(
              {superType, superType, superType}, {componentType},
              [&superType, &componentType,
               &loc](mlir::ValueRange args, mlir::OpBuilder &bldr, const mlir::TypeConverter &) {
    // Reference to self
    auto self = bldr.create<SelfOp>(loc, componentType);
    // Do the computation
    auto op = bldr.create<InRangeOp>(loc, superType, args[0], args[1], args[2]);
    // Store the result
    bldr.create<WriteFieldOp>(loc, self, "$super", op);
    // Return self
    bldr.create<mlir::func::ReturnOp>(loc, mlir::ValueRange({self}));
  }
          )
  ).build(builder, TC);
}

void addComponent(mlir::OpBuilder &builder, const mlir::TypeConverter &TC) {
  auto componentType = Component(builder);
  auto convertedComponentType = TC.convertType(componentType);

  llvm::dbgs() << "convertedComponentType = " << convertedComponentType << "\n";
  mlir::Location loc = getLocationForBuiltin(builder, componentType.getName().getValue());
  auto c =
      builtinCommon(
          ComponentBuilder()
              .name(componentType.getName().getValue())
              .location(loc)
              .fillBody(
                  {}, {convertedComponentType},
                  [&convertedComponentType,
                   &loc](mlir::ValueRange, mlir::OpBuilder &bldr, const mlir::TypeConverter &) {
    auto op = bldr.create<component::CreateStructOp>(loc, convertedComponentType);
    bldr.create<function::ReturnOp>(loc, mlir::ValueRange({op}));
  }
              )
      ).build(builder, TC);
  llvm::dbgs() << "component = " << c << "\n";
  llvm::dbgs() << "component.getType() = " << c.getType() << "\n";
}

void addNondetRegCommon(
    mlir::OpBuilder &builder, const mlir::TypeConverter &TC, mlir::StringRef name,
    ComponentLike superType
) {
  auto componentType = ComplexComponentType::get(builder.getContext(), name, superType, true);
  auto convertedSuperType = TC.convertType(superType);
  auto convertedComponentType = TC.convertType(componentType);
  mlir::Location loc = getLocationForBuiltin(builder, name);
  builtinCommon(
      ComponentBuilder()
          .name(name)
          .field("$super", convertedSuperType)
          .field("reg", convertedSuperType, /*isColumn=*/true)
          .location(loc)
          .constructorLocs({loc})
          .fillBody(
              {convertedSuperType}, {convertedComponentType},
              [&convertedComponentType,
               &loc](mlir::ValueRange args, mlir::OpBuilder &bldr, const mlir::TypeConverter &) {
    // Reference to self
    auto self = bldr.create<component::CreateStructOp>(loc, convertedComponentType);
    bldr.create<component::FieldWriteOp>(loc, self, "reg", args[0]);
    // Store the result
    bldr.create<component::FieldWriteOp>(loc, self, "$super", args[0]);
    // Return self
    bldr.create<function::ReturnOp>(loc, mlir::ValueRange(self));
  }
          )
  ).build(builder, TC);
}

void addNondetReg(mlir::OpBuilder &builder, const mlir::TypeConverter &TC) {
  addNondetRegCommon(builder, TC, "NondetReg", Val(builder));
}

void addNondetExtReg(mlir::OpBuilder &builder, const mlir::TypeConverter &TC) {
  addNondetRegCommon(builder, TC, "NondetExtReg", ExtVal(builder));
}

void addMakeExt(mlir::OpBuilder &builder, const mlir::TypeConverter &TC) {
  auto superType = ExtVal(builder);
  auto valType = Val(builder);
  auto componentType = ComplexComponentType::get(builder.getContext(), "MakeExt", superType, true);
  mlir::Location loc = getLocationForBuiltin(builder, componentType.getName().getValue());
  builtinCommon(
      ComponentBuilder()
          .name(componentType.getName().getValue())
          .field("$super", superType)
          .location(loc)
          .fillBody(
              {valType}, {componentType},
              [&superType, &componentType,
               &loc](mlir::ValueRange args, mlir::OpBuilder &bldr, const mlir::TypeConverter &) {
    // Reference to self
    auto self = bldr.create<SelfOp>(loc, componentType);
    auto ext = bldr.create<MakeExtOp>(loc, superType, args[0]);
    // Store the result
    bldr.create<WriteFieldOp>(loc, self, "$super", ext);
    // Return self
    bldr.create<mlir::func::ReturnOp>(loc, mlir::ValueRange({self}));
  }
          )
  ).build(builder, TC);
}

void addEqzExt(mlir::OpBuilder &builder, const mlir::TypeConverter &TC) {
  auto superType = Component(builder);
  auto valType = ExtVal(builder);

  auto componentType = ComplexComponentType::get(builder.getContext(), "EqzExt", superType, true);
  mlir::Location loc = getLocationForBuiltin(builder, componentType.getName().getValue());
  builtinCommon(
      ComponentBuilder()
          .name(componentType.getName().getValue())
          .field("$super", superType)
          .location(loc)
          .fillBody(
              {valType}, {componentType},
              [&superType, &componentType,
               &loc](mlir::ValueRange args, mlir::OpBuilder &bldr, const mlir::TypeConverter &) {
    // Reference to self
    auto self = bldr.create<SelfOp>(loc, componentType);
    bldr.create<EqzExtOp>(loc, args[0]);
    auto compCtorRef = bldr.create<ConstructorRefOp>(
        loc, superType.getName(), bldr.getFunctionType(mlir::TypeRange(), superType), true
    );
    auto comp = bldr.create<mlir::func::CallIndirectOp>(loc, compCtorRef, mlir::ValueRange());
    // Store the result
    bldr.create<WriteFieldOp>(loc, self, "$super", comp.getResult(0));
    // Return self
    bldr.create<mlir::func::ReturnOp>(loc, mlir::ValueRange({self}));
  }
          )
  ).build(builder, TC);
}

void addTrivial(
    mlir::OpBuilder &builder, const mlir::TypeConverter &TC, ComponentLike componentType
) {
  selfConstructs(
      builtinCommon(ComponentBuilder()
                        .name(componentType.getName().getValue())
                        .field("$super", TC.convertType(componentType))),
      TC.convertType(componentType), component::StructType::get(componentType.getName()),
      componentType.getName().getValue()
  )
      .build(builder, TC);
}

constexpr mlir::StringRef ARRAY_TYPE_PARAM = "T";
constexpr mlir::StringRef ARRAY_SIZE_PARAM = "N";

void addArrayComponent(mlir::OpBuilder &builder, const mlir::TypeConverter &tc) {
  auto typeParam = mlir::FlatSymbolRefAttr::get(builder.getStringAttr(ARRAY_TYPE_PARAM));
  auto innerType = llzk::polymorphic::TypeVarType::get(typeParam);
  auto sizeAttr = mlir::FlatSymbolRefAttr::get(builder.getStringAttr(ARRAY_SIZE_PARAM));
  auto componentType = Array(innerType, sizeAttr, tc);

  selfConstructs(
      builtinCommon(ComponentBuilder()
                        .name("Array")
                        .forceGeneric()
                        .typeParam(ARRAY_TYPE_PARAM)
                        .typeParam(ARRAY_SIZE_PARAM)
                        .field("$super", componentType)),
      componentType,
      component::StructType::get(
          componentType.getName(), builder.getArrayAttr({typeParam, sizeAttr})
      ),
      componentType.getName().getValue()
  )
      .build(builder, tc);
}
#define MAYBE(name) if (definedNames.find(name) == definedNames.end())

void zml::addBuiltinBindings(
    zhl::TypeBindings &bindings, const std::unordered_set<std::string_view> &definedNames
) {
  auto &Val = bindings.Get("Val");
  auto &ExtVal = bindings.Get("ExtVal");

  MAYBE("NondetReg") {
    bindings.CreateBuiltin(
        "NondetReg", Val, zhl::ParamsMap(), zhl::ParamsMap().declare("v", Val), zhl::MembersMap()
    );
  }
  MAYBE("NondetExtReg") {
    bindings.CreateBuiltin(
        "NondetExtReg", ExtVal, zhl::ParamsMap(), zhl::ParamsMap().declare("v", ExtVal),
        zhl::MembersMap()
    );
  }
  MAYBE("InRange") {
    bindings.CreateBuiltin(
        "InRange", Val, zhl::ParamsMap(),
        zhl::ParamsMap().declare("low", Val).declare("mid", Val).declare("high", Val),
        zhl::MembersMap()
    );
  }
  MAYBE("BitAnd") {
    bindings.CreateBuiltin(
        "BitAnd", Val, zhl::ParamsMap(), zhl::ParamsMap().declare("lhs", Val).declare("rhs", Val),
        zhl::MembersMap()
    );
  }
  MAYBE("Add") {
    bindings.CreateBuiltin(
        "Add", Val, zhl::ParamsMap(), zhl::ParamsMap().declare("lhs", Val).declare("rhs", Val),
        zhl::MembersMap()
    );
  }
  MAYBE("ExtAdd") {
    bindings.CreateBuiltin(
        "ExtAdd", ExtVal, zhl::ParamsMap(),
        zhl::ParamsMap().declare("lhs", ExtVal).declare("rhs", ExtVal), zhl::MembersMap()
    );
  }
  MAYBE("Sub") {
    bindings.CreateBuiltin(
        "Sub", Val, zhl::ParamsMap(), zhl::ParamsMap().declare("lhs", Val).declare("rhs", Val),
        zhl::MembersMap()
    );
  }
  MAYBE("ExtSub") {
    bindings.CreateBuiltin(
        "ExtSub", ExtVal, zhl::ParamsMap(),
        zhl::ParamsMap().declare("lhs", ExtVal).declare("rhs", ExtVal), zhl::MembersMap()
    );
  }
  MAYBE("Mul") {
    bindings.CreateBuiltin(
        "Mul", Val, zhl::ParamsMap(), zhl::ParamsMap().declare("lhs", Val).declare("rhs", Val),
        zhl::MembersMap()
    );
  }
  MAYBE("ExtMul") {
    bindings.CreateBuiltin(
        "ExtMul", ExtVal, zhl::ParamsMap(),
        zhl::ParamsMap().declare("lhs", ExtVal).declare("rhs", ExtVal), zhl::MembersMap()
    );
  }
  MAYBE("Mod") {
    bindings.CreateBuiltin(
        "Mod", Val, zhl::ParamsMap(), zhl::ParamsMap().declare("lhs", Val).declare("rhs", Val),
        zhl::MembersMap()
    );
  }
  MAYBE("Inv") {
    bindings.CreateBuiltin(
        "Inv", Val, zhl::ParamsMap(), zhl::ParamsMap().declare("v", Val), zhl::MembersMap()
    );
  }
  MAYBE("ExtInv") {
    bindings.CreateBuiltin(
        "ExtInv", ExtVal, zhl::ParamsMap(), zhl::ParamsMap().declare("v", ExtVal), zhl::MembersMap()
    );
  }
  MAYBE("Isz") {
    bindings.CreateBuiltin(
        "Isz", Val, zhl::ParamsMap(), zhl::ParamsMap().declare("v", Val), zhl::MembersMap()
    );
  }
  MAYBE("Neg") {
    bindings.CreateBuiltin(
        "Neg", Val, zhl::ParamsMap(), zhl::ParamsMap().declare("v", Val), zhl::MembersMap()
    );
  }
  MAYBE("MakeExt") {
    bindings.CreateBuiltin(
        "MakeExt", ExtVal, zhl::ParamsMap(), zhl::ParamsMap().declare("v", Val), zhl::MembersMap()
    );
  }
  MAYBE("EqzExt") {
    bindings.CreateBuiltin(
        "EqzExt", bindings.Component(), zhl::ParamsMap(), zhl::ParamsMap().declare("v", ExtVal),
        zhl::MembersMap()
    );
  }
}

/// Adds the builtin operations that have not been overriden
void zml::addBuiltins(
    mlir::OpBuilder &builder, const std::unordered_set<std::string_view> &definedNames,
    const mlir::TypeConverter &tc
) {
  assert(definedNames.find("Component") == definedNames.end() && "Can't redefine Component type");
  addComponent(builder, tc);

  assert(definedNames.find("Val") == definedNames.end() && "Can't redefine Val type");
  assert(definedNames.find("ExtVal") == definedNames.end() && "Can't redefine ExtVal type");
  assert(definedNames.find("String") == definedNames.end() && "Can't redefine String type");
  assert(definedNames.find("Array") == definedNames.end() && "Can't redefine Array type");
  // addTrivial(builder, tc, Val(builder));
  // addTrivial(builder, tc, ExtVal(builder));
  // addTrivial(builder, tc, String(builder));
  // addArrayComponent(builder, tc);

  MAYBE("NondetReg") { addNondetReg(builder, tc); }
  MAYBE("NondetExtReg") { addNondetExtReg(builder, tc); }
  // MAYBE("MakeExt") { addMakeExt(builder, tc); }
  // MAYBE("EqzExt") { addEqzExt(builder, tc); }
  // MAYBE("InRange") { addInRange(builder, tc); }
  // MAYBE("BitAnd") { addBinOp<BitAndOp>(builder, tc, "BitAnd"); }
  // MAYBE("Add") { addBinOp<AddOp>(builder, tc, "Add"); }
  // MAYBE("Sub") { addBinOp<SubOp>(builder, tc, "Sub"); }
  // MAYBE("Mul") { addBinOp<MulOp>(builder, tc, "Mul"); }
  // MAYBE("ExtAdd") { addExtBinOp<ExtAddOp>(builder, tc, "ExtAdd"); }
  // MAYBE("ExtSub") { addExtBinOp<ExtSubOp>(builder, tc, "ExtSub"); }
  // MAYBE("ExtMul") { addExtBinOp<ExtMulOp>(builder, tc, "ExtMul"); }
  // MAYBE("Mod") { addBinOp<ModOp>(builder, tc, "Mod"); }
  // MAYBE("Inv") { addUnaryOp<InvOp>(builder, tc, "Inv"); }
  // MAYBE("ExtInv") { addExtUnaryOp<ExtInvOp>(builder, tc, "ExtInv"); }
  // MAYBE("Isz") { addUnaryOp<IsZeroOp>(builder, tc, "Isz"); }
  // MAYBE("Neg") { addUnaryOp<NegOp>(builder, tc, "Neg"); }
}

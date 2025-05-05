//===- LLZKTypeConverter.cpp - LLZK type conversion from ZML ----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <iterator>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llzk/Dialect/Array/IR/Ops.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Bool/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Ops.h>
#include <llzk/Dialect/Felt/IR/Types.h>
#include <llzk/Dialect/Polymorphic/IR/Types.h>
#include <llzk/Dialect/String/IR/Types.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Types.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/Types.h>
#include <zklang/Passes/ConvertZmlToLlzk/LLZKTypeConverter.h>

#define DEBUG_TYPE "llzk-type-converter"

using namespace llzk;
using namespace llzk::array;
using namespace llzk::felt;
using namespace llzk::boolean;
using namespace mlir;

static std::optional<mlir::Value> unrealizedCastMaterialization(
    mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs, mlir::Location loc
) {

  assert(inputs.size() == 1);
  return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
}

static mlir::Type deduceArrayType(mlir::Attribute attr) {
  LLVM_DEBUG(llvm::dbgs() << "deduceArrayType(" << attr << ")\n");
  if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr)) {
    return typeAttr.getValue();
  }
  llvm_unreachable("Failed to convert array type");
  return nullptr;
}

template <typename Attr> static Attribute convert(Attr);

template <> Attribute convert(zml::ConstExprAttr attr) {
  return mlir::AffineMapAttr::get(attr.getMap());
}

template <> Attribute convert(zml::LiftedExprAttr attr) { return attr.getSymbol(); }

template <typename Attr> static Attribute pass(Attr a) { return a; }

static SmallVector<Attribute>
convertParamAttrs(ArrayRef<mlir::Attribute> in, LLZKTypeConverter &converter) {
  return llvm::map_to_vector(in, [&](mlir::Attribute attr) -> mlir::Attribute {
    return llvm::TypeSwitch<Attribute, Attribute>(attr)
        .Case([&converter](TypeAttr typeAttr) {
      return mlir::TypeAttr::get(converter.convertType(typeAttr.getValue()));
    })
        .Case(convert<zml::ConstExprAttr>)
        .Case(convert<zml::LiftedExprAttr>)
        .Default(pass<Attribute>);
  });
}

static mlir::Attribute getSizeAttr(mlir::Attribute attr) {
  LLVM_DEBUG(llvm::dbgs() << "getSizeSym(" << attr << ")\n");
  return llvm::TypeSwitch<Attribute, Attribute>(attr)
      .Case(pass<SymbolRefAttr>)
      .Case(pass<IntegerAttr>)
      .Case(convert<zml::ConstExprAttr>)
      .Case(convert<zml::LiftedExprAttr>)
      .Default([](auto) {
    llvm_unreachable("was expecting a symbol, number, or an affine expression");
    return nullptr;
  });
}

llzk::LLZKTypeConverter::LLZKTypeConverter(const ff::FieldData &Field)
    : field(Field),
      feltEquivalentTypes(
          {"Val", "Add", "Sub", "Mul", "BitAnd", "Inv", "Isz", "InRange", "Neg", "Mod"}
      ),
      extValBuiltins({"ExtVal", "ExtAdd", "ExtSub", "ExtMul", "ExtInv", "MakeExt"}) {

  addConversion([](mlir::Type t) { return t; });

  // Conversions from ZML to LLZK

  addConversion([&](zml::ComponentType t) -> mlir::Type {
    auto convertedAttrs = convertParamAttrs(t.getParams(), *this);
    return llzk::component::StructType::get(
        t.getName(), mlir::ArrayAttr::get(t.getContext(), convertedAttrs)
    );
  });

  addConversion([&](zml::ComponentType t) -> std::optional<mlir::Type> {
    if (extValBuiltins.find(t.getName().getValue()) != extValBuiltins.end() && t.getBuiltin()) {
      return createArrayRepr(t.getContext());
    }
    return std::nullopt;
  });

  addConversion([&](zml::ComponentType t) -> std::optional<mlir::Type> {
    if (t.getName().getValue() != "Array") {
      return std::nullopt;
    }
    LLVM_DEBUG(llvm::dbgs() << "addConversion: " << t << "\n");
    assert(t.getParams().size() == 2);
    auto typeAttr = t.getParams()[0];
    auto sizeAttr = t.getParams()[1];

    llvm::SmallVector<mlir::Attribute> dims({getSizeAttr(sizeAttr)});
    auto inner = convertType(deduceArrayType(typeAttr));
    if (auto innerArr = mlir::dyn_cast<llzk::array::ArrayType>(inner)) {
      auto innerDims = innerArr.getDimensionSizes();
      dims.insert(dims.end(), innerDims.begin(), innerDims.end());
      inner = innerArr.getElementType();
    }

    return llzk::array::ArrayType::get(inner, dims);
  });

  addConversion([](zml::ComponentType t) -> std::optional<mlir::Type> {
    if (t.getName().getValue() == "String") {
      return llzk::string::StringType::get(t.getContext());
    }
    return std::nullopt;
  });

  addConversion([&](zml::ComponentType t) -> std::optional<mlir::Type> {
    if (feltEquivalentTypes.find(t.getName().getValue()) != feltEquivalentTypes.end() &&
        t.getBuiltin()) {
      return llzk::felt::FeltType::get(t.getContext());
    }
    return std::nullopt;
  });

  addConversion([&](zml::VarArgsType t) {
    std::vector<int64_t> shape = {mlir::ShapedType::kDynamic};
    return llzk::array::ArrayType::get(convertType(t.getInner()), shape);
  });

  addConversion([](zml::TypeVarType t) {
    return llzk::polymorphic::TypeVarType::get(t.getContext(), t.getName());
  });

  addSourceMaterialization(unrealizedCastMaterialization);
  addTargetMaterialization(unrealizedCastMaterialization);
  addArgumentMaterialization(unrealizedCastMaterialization);
}

mlir::Type LLZKTypeConverter::createArrayRepr(mlir::MLIRContext *ctx) const {
  return llzk::array::ArrayType::get(
      llzk::felt::FeltType::get(ctx), {static_cast<long>(field.degree)}
  );
}

mlir::Value LLZKTypeConverter::collectValues(
    mlir::ValueRange vals, mlir::Location loc, mlir::OpBuilder &builder
) const {
  return builder.create<CreateArrayOp>(
      loc, cast<ArrayType>(createArrayRepr(builder.getContext())), vals
  );
}

using ValueWrap = zml::extval::BaseConverter::ValueWrap;
using Values = mlir::SmallVector<zml::extval::BaseConverter::ValueWrap>;

Values LLZKTypeConverter::wrapArrayValues(mlir::Value v, mlir::OpBuilder &builder) const {
  assertIsValidRepr(v);
  Values vals;
  vals.reserve(field.degree);
  for (uint64_t i = 0; i < field.degree; i++) {
    auto idx = builder.create<arith::ConstantIndexOp>(v.getLoc(), i);
    vals.push_back(
        ValueWrap(builder.create<ReadArrayOp>(v.getLoc(), v, ValueRange(idx)), builder, *this)
    );
  }

  return vals;
}

mlir::Value
LLZKTypeConverter::createAddOp(mlir::Value lhs, mlir::Value rhs, mlir::OpBuilder &builder) const {
  return builder.create<AddFeltOp>(lhs.getLoc(), lhs, rhs);
}

mlir::Value
LLZKTypeConverter::createSubOp(mlir::Value lhs, mlir::Value rhs, mlir::OpBuilder &builder) const {
  return builder.create<SubFeltOp>(lhs.getLoc(), lhs, rhs);
}

mlir::Value
LLZKTypeConverter::createMulOp(mlir::Value lhs, mlir::Value rhs, mlir::OpBuilder &builder) const {
  return builder.create<MulFeltOp>(lhs.getLoc(), lhs, rhs);
}

mlir::Value LLZKTypeConverter::createNegOp(mlir::Value v, mlir::OpBuilder &builder) const {
  return builder.create<NegFeltOp>(v.getLoc(), v);
}

mlir::Value LLZKTypeConverter::createInvOp(mlir::Value v, mlir::OpBuilder &builder) const {
  return builder.create<InvFeltOp>(v.getLoc(), v);
}

mlir::Value LLZKTypeConverter::createLitOp(uint64_t v, mlir::OpBuilder &builder) const {
  auto felt = FeltType::get(builder.getContext());
  return builder.create<FeltConstantOp>(
      builder.getUnknownLoc(), felt, FeltConstAttr::get(builder.getContext(), llvm::APInt(64, v))
  );
}

mlir::Value LLZKTypeConverter::createIszOp(mlir::Value v, mlir::OpBuilder &builder) const {
  auto zero = builder.create<FeltConstantOp>(
      v.getLoc(), FeltConstAttr::get(builder.getContext(), mlir::APInt::getZero(64))
  );
  return builder.create<CmpOp>(
      v.getLoc(), FeltCmpPredicateAttr::get(builder.getContext(), FeltCmpPredicate::EQ), v, zero
  );
}

mlir::Operation *LLZKTypeConverter::createAssertOp(
    mlir::Value cond, mlir::StringAttr msg, mlir::OpBuilder &builder
) const {
  return builder.create<AssertOp>(cond.getLoc(), cond, msg);
}

mlir::Value
LLZKTypeConverter::createAndOp(mlir::Value lhs, mlir::Value rhs, mlir::OpBuilder &builder) const {
  return builder.create<AndBoolOp>(lhs.getLoc(), lhs, rhs);
}

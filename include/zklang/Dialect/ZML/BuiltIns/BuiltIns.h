//===- BuiltIns.h - Builtins definitions ------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes definitions for builtins types and components in the
// zirgen DSL.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Builders.h>
#include <string_view>
#include <unordered_set>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>
#include <zklang/Dialect/ZML/IR/TypeInterfaces.h>

namespace mlir {
class MLIRContext;
class TypeConverter;
} // namespace mlir

namespace zml {

void addBuiltinBindings(zhl::TypeBindings &, const std::unordered_set<std::string_view> &);
// Add builtin components using the given builder
void addBuiltins(mlir::OpBuilder &, const std::unordered_set<std::string_view> &, const mlir::TypeConverter &);
mlir::Operation *getBuiltInOp(mlir::StringRef);

namespace builtins {

ComponentLike Component(mlir::MLIRContext *);

inline ComponentLike Component(mlir::OpBuilder &builder) { return Component(builder.getContext()); }

ComponentLike Val(mlir::MLIRContext *);

inline ComponentLike Val(mlir::OpBuilder &builder) { return Val(builder.getContext()); }

ComponentLike String(mlir::MLIRContext *);

inline ComponentLike String(mlir::OpBuilder &builder) { return String(builder.getContext()); }

ComponentLike ExtVal(mlir::MLIRContext *);

inline ComponentLike ExtVal(mlir::OpBuilder &builder) { return ExtVal(builder.getContext()); }

ComponentLike Array(mlir::Type, mlir::Attribute, const mlir::TypeConverter &);

} // namespace builtins

static const std::unordered_set<std::string> BuiltInComponentNames = {
    "BitAnd", "Add", "Sub",     "Mul",    "Inv",    "Isz",    "Neg",    "Val",     "String",
    "Array",  "Mod", "InRange", "ExtInv", "ExtAdd", "ExtSub", "ExtMul", "MakeExt", "ExtVal"
};

static const std::unordered_set<std::string_view> BuiltinsDontNeedAlloc = {
    "Add", "Sub", "Mul", "Isz", "Neg", "Mod", "InRange", "ExtAdd", "ExtSub", "ExtMul", "MakeExt"
};

inline bool isBuiltinDontNeedAlloc(std::string_view name) {
  return BuiltinsDontNeedAlloc.find(name) != BuiltinsDontNeedAlloc.end();
}

static const char BitAndStr[] = "BitAnd";
static const char AddStr[] = "Add";
static const char SubStr[] = "Sub";
static const char MulStr[] = "Mul";
static const char InvStr[] = "Inv";
static const char ExtAddStr[] = "ExtAdd";
static const char ExtSubStr[] = "ExtSub";
static const char ExtMulStr[] = "ExtMul";
static const char ExtInvStr[] = "ExtInv";
static const char MakeExtStr[] = "MakeExt";
static const char EqzExtStr[] = "EqzExt";
static const char IszStr[] = "Isz";
static const char NegStr[] = "Neg";
static const char ValStr[] = "Val";
static const char ExtValStr[] = "ExtVal";
static const char StrStr[] = "String";
static const char ComponentStr[] = "Component";
static const char ArrayStr[] = "Array";
static const char ModStr[] = "Mod";
static const char InRangeStr[] = "InRange";

// Taken from zirgen
// Builtins that are defined using the DSL.
static llvm::StringLiteral zirPreamble = R"(

component Reg(v: Val) {
   reg := NondetReg(v);
   v = reg;
   reg
}

component ExtReg(v: ExtVal) {
   reg := NondetExtReg(v);
   EqzExt(ExtSub(reg, v));
   reg
}

function Div(lhs: Val, rhs: Val) {
   reciprocal := Inv(rhs);
   reciprocal * rhs = 1;

   reciprocal * lhs
}

extern Log(message: String, vals: Val...);
extern Abort();
extern Assert(x: Val, message: String);



)";

} // namespace zml

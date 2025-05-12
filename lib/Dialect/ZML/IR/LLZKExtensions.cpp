//===- LLZKExtensions.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file defines extensions to llzk dialects that enable them to interact
// with ZML entities.
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/StringRef.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Felt/IR/Types.h>
#include <llzk/Dialect/String/IR/Types.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/IR/OpInterfaces.h>
#include <zklang/Dialect/ZML/IR/TypeInterfaces.h>
#include <zklang/Dialect/ZML/IR/Types.h>

using namespace mlir;
using namespace llzk::component;
using namespace llzk::felt;
using namespace llzk::array;
using namespace llzk::string;
using namespace zml;

namespace {

struct StructDefDefinesBodyFuncInterfaceImpl
    : public DefinesBodyFunc::ExternalModel<StructDefDefinesBodyFuncInterfaceImpl, StructDefOp> {
  StringRef getBodyFuncName(Operation *) const { return "compute"; }
};

struct StructDefDefinesConstrainFuncInterfaceImpl
    : public DefinesConstrainFunc::ExternalModel<
          StructDefDefinesConstrainFuncInterfaceImpl, StructDefOp> {
  StringRef getConstrainFuncName(Operation *) const { return "constrain"; }
};

struct StructDefComponentInterfaceImpl
    : public ComponentInterface::ExternalModel<StructDefComponentInterfaceImpl, StructDefOp> {
  Type getType(Operation *) const {
    assert(false && "TODO");
    return nullptr;
  }
  FailureOr<Type> getSuperType(Operation *) const {
    assert(false && "TODO");
    return failure();
  }
  FailureOr<Type> lookupFieldType(Operation *, FlatSymbolRefAttr) const {
    assert(false && "TODO");
    return failure();
  }
  bool getBuiltin(Operation *) const {
    assert(false && "TODO");
    return false;
  }
  bool getUsesBackVariables(Operation *) const {
    assert(false && "TODO");
    return false;
  }
  bool isRoot(Operation *op) const { return isRootImpl(mlir::cast<StructDefOp>(op)); }
  Region &getRegion(Operation *) const {
    assert(false && "TODO");
    llvm_unreachable("TODO");
  }
  bool hasUnifiedBody(Operation *) const {
    assert(false && "TODO");
    return false;
  }
  FlatSymbolRefAttr getSuperFieldName(Operation *) const {
    assert(false && "TODO");
    return nullptr;
  }

private:
  bool isRootImpl(StructDefOp op) const { return op.getName() == "Component"; }
};

struct FeltComponentLikeImpl
    : public ComponentLike::ExternalModel<FeltComponentLikeImpl, FeltType> {

  FlatSymbolRefAttr getName(Type type) const {
    return FlatSymbolRefAttr::get(StringAttr::get(type.getContext(), "Val"));
  }
  // Type getSuperType(Type type) const { return RootType::get(type.getContext()); }
  ArrayRef<Attribute> getParams(Type) const { return {}; }
  bool getBuiltin(Type) const { return true; }

  ComponentInterface getDefinition(Type, SymbolTableCollection &, Operation *) const {
    assert(false && "TODO");
    return nullptr;
  }
  bool isRoot(Type) const { return false; }
  bool isConcreteArray(Type) const { return false; }
  bool isArray(Type) const { return false; }
  FailureOr<Type> getFirstMatchingSuperType(Type, llvm::function_ref<bool(Type)>) const;
  FailureOr<Type> getArrayInnerType(Type) const { return failure(); }
  FailureOr<Attribute> getArraySize(Type) const { return failure(); }
  ComponentLike getSuperType(Type t) const { return RootType::get(t.getContext()); }
  bool subtypeOf(Type, Type t) const { return isa<FeltType, RootType>(t); }
};

struct StringComponentLikeImpl
    : public ComponentLike::ExternalModel<StringComponentLikeImpl, StringType> {

  FlatSymbolRefAttr getName(Type type) const {
    return FlatSymbolRefAttr::get(StringAttr::get(type.getContext(), "String"));
  }
  // Type getSuperType(Type type) const { return RootType::get(type.getContext()); }
  ArrayRef<Attribute> getParams(Type) const { return {}; }
  bool getBuiltin(Type) const { return true; }

  ComponentInterface getDefinition(Type, SymbolTableCollection &, Operation *) const {
    assert(false && "TODO");
    return nullptr;
  }
  bool isRoot(Type) const { return false; }
  bool isConcreteArray(Type) const { return false; }
  bool isArray(Type) const { return false; }
  FailureOr<Type> getFirstMatchingSuperType(Type, llvm::function_ref<bool(Type)>) const;
  FailureOr<Type> getArrayInnerType(Type) const { return failure(); }
  FailureOr<Attribute> getArraySize(Type) const { return failure(); }
  ComponentLike getSuperType(Type t) const { return RootType::get(t.getContext()); }
  bool subtypeOf(Type, Type t) const { return isa<StringType, RootType>(t); }
};

struct ArrayComponentLikeImpl
    : public ComponentLike::ExternalModel<ArrayComponentLikeImpl, ArrayType> {

  FlatSymbolRefAttr getName(Type type) const {
    return FlatSymbolRefAttr::get(StringAttr::get(type.getContext(), "Array"));
  }
  // Type getSuperType(Type type) const { return RootType::get(type.getContext()); }
  ArrayRef<Attribute> getParams(Type t) const {
    auto arr = mlir::cast<ArrayType>(t);
    // Since zirgen arrays are different from llzk arrays we need to do a bit of different logic.
    assert(false && "TODO");
    return {};
  }
  bool getBuiltin(Type) const { return true; }

  ComponentInterface getDefinition(Type, SymbolTableCollection &, Operation *) const {
    assert(false && "TODO");
    return nullptr;
  }
  bool isRoot(Type) const { return false; }
  bool isConcreteArray(Type) const { return true; }
  bool isArray(Type) const { return true; }
  FailureOr<Type> getFirstMatchingSuperType(Type, llvm::function_ref<bool(Type)>) const;
  FailureOr<Type> getArrayInnerType(Type) const {
    assert(false && "TODO");
    return failure();
  }
  FailureOr<Attribute> getArraySize(Type) const {
    assert(false && "TODO");
    return failure();
  }
  ComponentLike getSuperType(Type t) const { return RootType::get(t.getContext()); }
  bool subtypeOf(Type, Type t) const { return isa<ArrayType, RootType>(t); }
};

struct TypeVarComponentLikeImpl
    : public ComponentLike::ExternalModel<TypeVarComponentLikeImpl, TypeVarType> {

  FlatSymbolRefAttr getName(Type type) const {
    return FlatSymbolRefAttr::get(StringAttr::get(type.getContext(), "String"));
  }
  // Type getSuperType(Type type) const { return RootType::get(type.getContext()); }
  ArrayRef<Attribute> getParams(Type) const { return {}; }
  bool getBuiltin(Type) const { return true; }

  ComponentInterface getDefinition(Type, SymbolTableCollection &, Operation *) const {
    assert(false && "TODO");
    return nullptr;
  }
  bool isRoot(Type) const { return false; }
  bool isConcreteArray(Type) const { return false; }
  bool isArray(Type) const { return false; }
  FailureOr<Type> getFirstMatchingSuperType(Type, llvm::function_ref<bool(Type)>) const;
  FailureOr<Type> getArrayInnerType(Type) const { return failure(); }
  FailureOr<Attribute> getArraySize(Type) const { return failure(); }
  ComponentLike getSuperType(Type t) const { return RootType::get(t.getContext()); }
  bool subtypeOf(Type, Type t) const { return isa<StringType, RootType>(t); }
};

} // namespace

void zml::loadLLZKDialectExtensions(MLIRContext &context) {
  StructDefOp::attachInterface<StructDefDefinesBodyFuncInterfaceImpl>(context);
  StructDefOp::attachInterface<StructDefDefinesConstrainFuncInterfaceImpl>(context);
  StructDefOp::attachInterface<StructDefComponentInterfaceImpl>(context);
  FeltType::attachInterface<FeltComponentLikeImpl>(context);
  StringType::attachInterface<StringComponentLikeImpl>(context);
  ArrayType::attachInterface<ArrayComponentLikeImpl>(context);
  TypeVarType::attachInterface<TypeVarComponentLikeImpl>(context);
}

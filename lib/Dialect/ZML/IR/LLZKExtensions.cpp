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
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZML/BuiltIns/BuiltIns.h>
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

  /// Returns the type this struct defines. Needs to return a ComponentLike compatible type.
  Type getType(Operation *op) const { return getTypeImpl(llvm::cast<StructDefOp>(op)); }

  FailureOr<Type> getSuperType(Operation *op) const {
    return getSuperTypeImpl(llvm::cast<StructDefOp>(op));
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
  FlatSymbolRefAttr getSuperFieldName(Operation *op) const {
    return FlatSymbolRefAttr::get(getSuperFieldNameImpl(mlir::cast<StructDefOp>(op)));
  }

private:
  Type getTypeImpl(StructDefOp op) const {
    llvm::dbgs() << "getTypeImpl for " << op.getName() << "\n";
    if (isRootImpl(op)) {
      llvm::dbgs() << "is the root component\n";
      return zml::builtins::Component(op.getContext());
    }

    Builder builder(op.getContext());
    auto superType = getSuperTypeImpl(op);
    if (failed(superType)) {
      llvm::dbgs() << "super type extraction failed!!!!\n";
    }
    llvm::dbgs() << "creating component\n";
    auto name = FlatSymbolRefAttr::get(builder.getStringAttr(op.getName()));
    llvm::dbgs() << "  with name = " << name << "\n";
    auto super = mlir::cast<ComponentLike>(*superType);
    llvm::dbgs() << "  with super = " << super << "\n";
    auto params = op.getConstParams().value_or(builder.getArrayAttr({}));
    llvm::dbgs() << "  with params = " << params << "\n";
    return ComplexComponentType::get(
        name, super, params.getValue(), op->getAttr("zml.builtin") != nullptr
    );
  }

  FailureOr<Type> getSuperTypeImpl(StructDefOp op) const {
    auto name = getSuperFieldNameImpl(op);
    auto field = op.getFieldDef(name);
    if (!field) {
      return op->emitError() << "field '" << name << "' not found in struct '" << op.getName()
                             << "'";
    }

    if (mlir::isa<ComponentLike>(field.getType())) {
      return field.getType();
    }

    if (auto structType = mlir::dyn_cast<StructType>(field.getType())) {
      auto mod = op.getParentOp();
      SymbolTableCollection st;
      auto def = structType.getDefinition(st, mod);
      if (failed(def)) {
        return field.emitError() << "definition of field type " << structType << " not found";
      }

      return getTypeImpl(**def);
    }
    return field.emitError() << "field type " << field.getType()
                             << " is not supported by the ComponentInterface interface";
  }

  bool isRootImpl(StructDefOp op) const { return op.getName() == "Component"; }

  StringAttr getSuperFieldNameImpl(StructDefOp op) const {
    Builder builder(op.getContext());
    return builder.getStringAttr("$super");
  }
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

  ArrayRef<Attribute> getParams(Type t) const {
    auto arr = mlir::cast<ArrayType>(t);
    // Since zirgen arrays are different from llzk arrays we need to do a bit of different logic.
    Type inner;
    if (arr.getDimensionSizes().size() == 1) {
      inner = arr.getElementType();
    } else {
      inner = ArrayType::get(arr.getElementType(), arr.getDimensionSizes().drop_front());
    }
    Builder builder(t.getContext());
    return builder.getArrayAttr({TypeAttr::get(inner), arr.getDimensionSizes().front()});
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

//===- Dialect.cpp - ZML Dialect --------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Felt/IR/Types.h>
#include <llzk/Dialect/Polymorphic/IR/Types.h>
#include <llzk/Dialect/String/IR/Types.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>
#include <zklang/Dialect/ZML/IR/Attrs.h> // IWYU pragma: keep
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/IR/OpInterfaces.h>
#include <zklang/Dialect/ZML/IR/Ops.h> // IWYU pragma: keep
#include <zklang/Dialect/ZML/IR/TypeInterfaces.h>
#include <zklang/Dialect/ZML/Typing/Materialize.h>

// TableGen'd implementation files
#include <zklang/Dialect/ZML/IR/Dialect.cpp.inc>

template <> struct mlir::FieldParser<zhl::TypeBinding> {
  static FailureOr<zhl::TypeBinding> parse(AsmParser &parser) { return mlir::failure(); }
};

// Need a complete declaration of storage classes
#define GET_TYPEDEF_CLASSES
#include <zklang/Dialect/ZML/IR/Types.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <zklang/Dialect/ZML/IR/Attrs.cpp.inc>

using namespace zml;
using namespace mlir;

auto ZMLDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "zklang/Dialect/ZML/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "zklang/Dialect/ZML/IR/Types.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "zklang/Dialect/ZML/IR/Attrs.cpp.inc"
  >();
  // clang-format on

  declarePromisedInterface<llzk::component::StructDefOp, DefinesBodyFunc>();
  declarePromisedInterface<llzk::component::StructDefOp, DefinesConstrainFunc>();
  declarePromisedInterface<llzk::component::StructDefOp, ComponentInterface>();
  declarePromisedInterface<llzk::felt::FeltType, ComponentLike>();
  declarePromisedInterface<llzk::string::StringType, ComponentLike>();
  declarePromisedInterface<llzk::array::ArrayType, ComponentLike>();
  declarePromisedInterface<llzk::polymorphic::TypeVarType, ComponentLike>();
}

//===----------------------------------------------------------------------===//
// TypeBindingAttr
//===----------------------------------------------------------------------===//

Type TypeBindingAttr::materializeType() const {
  return materializeTypeBinding(getContext(), *getTypeBinding());
}

Type TypeBindingAttr::materializeType(const mlir::TypeConverter *tc) const {
  assert(tc);
  return materializeType(*tc);
}

Type TypeBindingAttr::materializeType(const mlir::TypeConverter &tc) const {
  return tc.convertType(materializeType());
}

FunctionType TypeBindingAttr::materializeCtorType() const {
  Builder b(getContext());
  return materializeTypeBindingConstructor(b, *getTypeBinding(), getTypeBinding()->getContext());
}

TypeBindingAttr TypeBindingAttr::getSuperTypeAttr() const {
  if (!getTypeBinding()->hasSuperType()) {
    return nullptr;
  }

  return TypeBindingAttr::get(getContext(), getTypeBinding()->getSuperType());
}

FailureOr<TypeBindingAttr>
TypeBindingAttr::getArrayElementAttr(llvm::function_ref<mlir::InFlightDiagnostic()> emitError
) const {
  auto elt = getTypeBinding()->getArrayElement(emitError);
  if (failed(elt)) {
    return mlir::failure();
  }

  return TypeBindingAttr::get(getContext(), *elt);
}

FailureOr<TypeBindingAttr>
TypeBindingAttr::getArraySizeAttr(llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  auto size = getTypeBinding()->getArraySize(emitError);
  if (failed(size)) {
    return mlir::failure();
  }

  return TypeBindingAttr::get(getContext(), *size);
}

TypeBindingAttr TypeBindingAttr::get(Operation *op, StringRef key) {
  return op->getAttrOfType<zml::TypeBindingAttr>(key);
}

void TypeBindingAttr::bind(Operation *op, StringRef key, bool overwrite) const {
  if (!overwrite) {
    assert(!get(op, key) && "overwriting type binding");
  }

  op->setDiscardableAttr(key, *this);
}

static StringRef argKey(BlockArgument &arg, SmallVectorImpl<char> &sto) {
  sto.clear();
  ("zml.arg_binding." + Twine(arg.getArgNumber())).toVector(sto);
  return StringRef(sto.data(), sto.size());
}

TypeBindingAttr TypeBindingAttr::get(Value val) {
  return TypeSwitch<Value, TypeBindingAttr>(val)
      .Case([](OpResult res) { return get(res.getDefiningOp()); })
      .Case([](BlockArgument arg) {
    SmallString<30> key;
    return get(arg.getParentBlock()->getParentOp(), argKey(arg, key));
  }).Default([](auto) { return nullptr; });
}

void TypeBindingAttr::bind(Value val, bool overwrite) const {
  return TypeSwitch<Value>(val)
      .Case([this, overwrite](OpResult res) { bind(res.getDefiningOp(), overwrite); })
      .Case([this, overwrite](BlockArgument arg) {
    SmallString<30> key;
    bind(arg.getParentBlock()->getParentOp(), argKey(arg, key), overwrite);
  }).Default([](auto) {});
}

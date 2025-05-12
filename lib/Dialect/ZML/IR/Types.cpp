//===- Types.cpp - ZML types ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <functional>
#include <llvm/ADT/SmallVectorExtras.h>
#include <zklang/Dialect/ZML/IR/Types.h>

#include <algorithm>
#include <iterator>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/TypeInterfaces.h>

using namespace mlir;

namespace zml {

bool isValidZMLType(mlir::Type type) {
  return llvm::isa<TypeVarType>(type) || llvm::isa<ComponentLike>(type) ||
         (llvm::isa<VarArgsType>(type) && isValidZMLType(llvm::cast<VarArgsType>(type).getInner()));
}

// inline mlir::LogicalResult
// checkValidZMLType(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type type) {
//   if (!isValidZMLType(type)) {
//     return emitError() << "expected " << "a valid ZML type" << " but found " << type;
//   } else {
//     return mlir::success();
//   }
// }

mlir::LogicalResult
checkValidParam(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Attribute attr) {
  if (llvm::isa<
          mlir::SymbolRefAttr, mlir::IntegerAttr, mlir::TypeAttr, ConstExprAttr, LiftedExprAttr>(
          attr
      )) {
    return mlir::success();
  }
  return emitError() << "expected either a symbol or a literal integer but got " << attr;
}

mlir::LogicalResult checkValidTypeParam(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Attribute attr
) {
  // TODO: Check for types passed as attributes.
  if (!llvm::isa<mlir::SymbolRefAttr>(attr)) {
    return emitError() << "expected a symbol but got " << attr;
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult ComplexComponentType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, ::mlir::FlatSymbolRefAttr compName,
    zml::ComponentLike superType, ::llvm::ArrayRef<::mlir::Attribute> params, bool
) {
  if (!superType) {
    return emitError() << "malformed IR: super type for " << compName << " cannot be null";
  }

  return mlir::failure(llvm::any_of(
      llvm::map_to_vector(params, std::bind_front(checkValidParam, emitError)), mlir::failed
  ));
}

ComponentInterface ComplexComponentType::getDefinition(
    ::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op
) {
  return mlir::dyn_cast_if_present<ComponentInterface>(
      symbolTable.lookupNearestSymbolFrom(op, getName())
  );
}

// FailureOr<Type> ComplexComponentType::getFirstMatchingSuperType(llvm::function_ref<bool(Type)>
// pred ) const {
//   if (pred(*this)) {
//     return *this;
//   }
//   if (auto super = getSuperType()) {
//     if (auto superComp = mlir::dyn_cast<ComplexComponentType>(super)) {
//       return superComp.getFirstMatchingSuperType(pred);
//     } else if (pred(super)) {
//       return super;
//     }
//   }
//   return failure();
// }

FailureOr<Attribute> ComplexComponentType::getArraySize() const {
  return getSuperType().getArraySize();
}

FailureOr<Type> ComplexComponentType::getArrayInnerType() const {
  return getSuperType().getArrayInnerType();
}

static bool equivalentTypeAttr(TypeAttr lhsType, TypeAttr rhsType) {
  auto lhsComp = mlir::dyn_cast_if_present<ComponentLike>(lhsType.getValue());
  auto rhsComp = mlir::dyn_cast_if_present<ComponentLike>(rhsType.getValue());

  return lhsComp && rhsComp && lhsComp.subtypeOf(rhsComp);
}

template <typename T> static bool neitherAre(Attribute lhs, Attribute rhs) {
  return !mlir::isa<T>(lhs) && !mlir::isa<T>(rhs);
}

static uint8_t priority(Attribute attr) {
  return llvm::TypeSwitch<Attribute, uint8_t>(attr).Case([](IntegerAttr) { return 3; }
  ).Case([](ConstExprAttr) {
    return 2;
  }).Case([](LiftedExprAttr) {
    return 2;
  }).Case([](SymbolRefAttr) {
    return 1;
  }).Default([](auto) { return 0; });
}

static bool checkSubtypeViaConcreteness(Attribute lhsAttr, Attribute rhsAttr) {
  auto lhs = priority(lhsAttr);
  auto rhs = priority(rhsAttr);
  return lhs && rhs && lhs >= rhs;
}

static bool equivalentParam(Attribute lhs, Attribute rhs) {
  if (lhs == rhs) {
    return true;
  }

  auto lhsType = mlir::dyn_cast_if_present<TypeAttr>(lhs);
  auto rhsType = mlir::dyn_cast_if_present<TypeAttr>(rhs);

  if (lhsType && rhsType) {
    return equivalentTypeAttr(lhsType, rhsType);
  }

  // Check that we are not in the situation where one side is a TypeAttr and the other isn't
  if (!neitherAre<TypeAttr>(lhs, rhs)) {
    return false;
  }

  return checkSubtypeViaConcreteness(lhs, rhs);
}

/// A type T may be a subtype of a type T' if for all parameters that are types p_0 and p_0', p_0 is
/// a subtype of p_0' and T and T' have the same name.
static bool subtypeViaParams(const ComponentLike &lhs, ComponentLike rhs) {
  // If they have the same name and number of parameters
  bool sameName = lhs.getName() == rhs.getName();
  bool sameNParams = lhs.getParams().size() == rhs.getParams().size();

  if (!(sameName && sameNParams)) {
    return false;
  }

  for (auto [lhsParam, rhsParam] : llvm::zip_equal(lhs.getParams(), rhs.getParams())) {
    if (!equivalentParam(lhsParam, rhsParam)) {
      return false;
    }
  }
  return true;
}

static bool subtypeViaParams(const ComponentLike &lhs, Type rhs) {
  if (auto rhsAsComp = mlir::dyn_cast_if_present<ComponentLike>(rhs)) {
    return subtypeViaParams(lhs, rhsAsComp);
  }
  return false;
}

bool ComplexComponentType::subtypeOf(Type other) const {
  if (*this == other || subtypeViaParams(*this, other)) {
    return true;
  }
  return getSuperType().subtypeOf(other);
}

FailureOr<ComponentLike>
ComponentLike::getFirstMatchingSuperType(llvm::function_ref<bool(ComponentLike)> pred) const {
  if (pred(*this)) {
    return *this;
  }
  if (auto super = getSuperType()) {
    return super.getFirstMatchingSuperType(pred);
  }
  return failure();
}

} // namespace zml

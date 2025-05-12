//===- TypeInterfaces.td - ZML type interfaces -------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the definitions of interfaces for types used by the
// ZML dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ZML_TYPE_INTERFACES
#define ZML_TYPE_INTERFACES

include "mlir/IR/Interfaces.td"

def ComponentLike : TypeInterface<"ComponentLike"> {
  let description = "Interface for types that can behave like a zirgen type";
  let cppNamespace = "::zml";

  let methods = [
    InterfaceMethod<"Returns the name of the component",
                    "::mlir::FlatSymbolRefAttr", "getName", (ins)>,
    InterfaceMethod<"Returns the super type of the component",
                    "::zml::ComponentLike", "getSuperType", (ins)>,
    InterfaceMethod<"Returns the generic parameters",
                    "::mlir::ArrayRef<::mlir::Attribute>", "getParams", (ins)>,
    InterfaceMethod<"Returns wether the type is a builtin or not", "bool",
                    "getBuiltin", (ins)>,
    InterfaceMethod<"Returns the op that defines this component type",
                    "::zml::ComponentInterface", "getDefinition",
                    (ins "::mlir::SymbolTableCollection&"
                     : $symbolTable, "::mlir::Operation *"
                     : $op)>,
    InterfaceMethod<"Returns wether the component associated to this type is "
                    "the root 'Component'",
                    "bool", "isRoot", (ins)>,
    InterfaceMethod<"Returns wether the component represents the 'Array' type",
                    "bool", "isConcreteArray", (ins)>,
    // InterfaceMethod<"Returns wether the component can behave like an
    // 'Array'",
    //                 "bool", "isArray", (ins)>,
    // InterfaceMethod<
    //     "Climbs the chain and returns the first super type (including self) "
    //     "that matches the predicate, or failure() otherwise",
    //     "::mlir::FailureOr<::mlir::Type>", "getFirstMatchingSuperType",
    //     (ins "::llvm::function_ref<bool(::mlir::Type)>"
    //      : $pred)>,
    InterfaceMethod<"Returns the inner type of 'Array'. If the type cannot "
                    "behave like 'Array' returns failure() instead",
                    "::mlir::FailureOr<::mlir::Type>", "getArrayInnerType",
                    (ins)>,
    InterfaceMethod<"Returns the size of 'Array'. If the type cannot behave "
                    "like 'Array' returns failure() instead",
                    "::mlir::FailureOr<::mlir::Attribute>", "getArraySize",
                    (ins)>,
    // InterfaceMethod<"Returns the super type as a ComponentLike. Returns "
    //                 "nullptr if the type does not meet the interface",
    //                 "::zml::ComponentLike", "getSuperTypeAsComp", (ins)>,
    InterfaceMethod<
        "Returns true if this type is a subtype of the given component", "bool",
        "subtypeOf",
        (ins "::zml::ComponentLike"
         : $type)>
  ];

  let extraClassDeclaration = [{
    /// Climbs the chain and returns the first super type (including self) 
    /// that matches the predicate, or failure() otherwise.
    ::mlir::FailureOr<::zml::ComponentLike> getFirstMatchingSuperType(::llvm::function_ref<bool(::zml::ComponentLike)>) const;

    /// Returns wether the component can behave like an 'Array'.
  bool isArray() const {
    return isConcreteArray() ||
        (getSuperType() && getSuperType().isArray());
  }

    /// Returns true if this type is a subtype of the given type
    bool subtypeOf(::mlir::Type t) const {
      if (auto c = ::mlir::dyn_cast_if_present<ComponentLike>(t)) {
        return subtypeOf(c);
      }
      return false;
    }

  }];
}

#endif

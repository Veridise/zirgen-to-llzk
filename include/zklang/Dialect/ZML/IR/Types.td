//===- Types.td - ZML type definitions ---------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the definitions of the types used by ZML operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "Dialect.td"
include "Attrs.td"
include "mlir/IR/SymbolInterfaces.td"

class ZMLType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ZMLDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TypeVar : ZMLType<"TypeVar", "tvar"> {
  let summary = "parametric type";

  let parameters = (ins "::mlir::FlatSymbolRefAttr":$name);
  let assemblyFormat = [{ `<` $name `>` }];

  let extraClassDeclaration = [{
    // Verifies that this type references a valid type parameter, relative to the given `op`.
    ::mlir::LogicalResult verifySymbol(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);
  }];
}

// XXX: Improvement idea: Make this a type interface and split the logic of the
// "special" types in different types that implement the interface.
def Component : ZMLType<"Component", "comp"> {
  let summary = "component type";
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$name,
      OptionalParameter<"mlir::Type">:$superType,
      OptionalArrayRefParameter<"::mlir::Attribute">:$params, "bool":$builtin);
  let assemblyFormat = [{
   `<` $name $builtin (`<` $params^ `>`)?(`:` $superType^)?  `>` }];

  let extraClassDeclaration = [{

    /// Gets the `component` op that defines this componnet. Provided `op` is
    /// used as a starting point for the lookup. Should not be assumed to
    /// be non-`null` as we don't verify all types during verification.
    ::zml::ComponentInterface getDefinition(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);
  
    // Verifies that this type references a valid component, relative to the given `op`.
    ::mlir::LogicalResult verifySymbol(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);

    /// Returns wether the component associated to this type is the root 'Component'
    bool isRoot() const { return getName().getValue() == "Component"; }
    bool isConcreteArray() const { return getName().getValue() == "Array"; }
    bool isArray() const { return isConcreteArray() || (getSuperTypeAsComp() != nullptr && getSuperTypeAsComp().isArray()); }
    /// Climbs the chain and returns the first super type (including self) that matches the predicate, or failure() otherwise.
    mlir::FailureOr<mlir::Type> getFirstMatchingSuperType(llvm::function_ref<bool(mlir::Type)>) const;

    mlir::FailureOr<mlir::Type> getArrayInnerType() const;
    mlir::FailureOr<mlir::Attribute> getArraySize() const;
    ComponentType getSuperTypeAsComp() const;
    bool subtypeOf(mlir::Type) const;

    static ComponentType getRoot(mlir::MLIRContext *ctx) { return ComponentType::get(ctx); }
    static ComponentType Component(mlir::MLIRContext *ctx) { return ComponentType::get(ctx); }
    static ComponentType Val(mlir::MLIRContext *ctx) { return ComponentType::get(ctx, "Val", Component(ctx), true); }
    static ComponentType ExtVal(mlir::MLIRContext *ctx) { return ComponentType::get(ctx, "ExtVal", Component(ctx), true); }
    static ComponentType String(mlir::MLIRContext *ctx) { return ComponentType::get(ctx, "String", Component(ctx), true); }
    static ComponentType Array(mlir::MLIRContext *ctx) { 
      auto symT = mlir::SymbolRefAttr::get(mlir::StringAttr::get(ctx, "T"));
      auto sizeVar = mlir::SymbolRefAttr::get(mlir::StringAttr::get(ctx, "N"));
      return ComponentType::get(ctx, "Array", Component(ctx), {symT, sizeVar}, true);
    }
    static ComponentType Array(mlir::MLIRContext *ctx, Type inner, uint64_t size) { 
      auto innerAttr = mlir::TypeAttr::get(inner);
      auto sizeAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), size);
      return ComponentType::get(ctx, "Array", Component(ctx), {innerAttr, sizeAttr}, true);
    }
  }];

  // TODO: Make this builders infer context since they already take a Type
  let builders = [TypeBuilder<(ins "::llvm::StringRef":$name, "Type":$superType,
                                  "::llvm::ArrayRef<::mlir::Attribute>":$params,
                                  "bool":$builtin),
                              [{
      return Base::get($_ctxt, mlir::SymbolRefAttr::get(mlir::StringAttr::get($_ctxt, name)), superType, params, builtin);
  }]>,
                  TypeBuilder<(ins), [{
      return Base::get($_ctxt, mlir::SymbolRefAttr::get(mlir::StringAttr::get($_ctxt, "Component")), nullptr, std::nullopt, true);
  }]>,
                  TypeBuilder<(ins "::llvm::StringRef":$name, "Type":$superType,
                                  "bool":$builtin),
                              [{
      return Base::get($_ctxt, mlir::SymbolRefAttr::get(mlir::StringAttr::get($_ctxt, name)), superType, std::nullopt, builtin);
    }]>];

  let genVerifyDecl = 1;
}

def VarArgs : ZMLType<"VarArgs", "vargs"> {
  let summary = "variable number of arguments";

  let parameters = (ins "::mlir::Type":$inner);
  let assemblyFormat = "`<` $inner `>`";
}

def AnyZMLType : AnyTypeOf<[TypeVar, Component, VarArgs, Index],
                           "type that can participate in ZML components">;

class TypeVarOr<Type type> : AnyTypeOf<[type, TypeVar]>;

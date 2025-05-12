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
include "TypeInterfaces.td"

class ZMLType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ZMLDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TypeVar : ZMLType<"TypeVar", "tvar"> {
  let summary = "parametric type";

  let parameters = (ins "::mlir::FlatSymbolRefAttr" : $name);
  let assemblyFormat = [{ `<` $name `>` }];

  let extraClassDeclaration = [{
    // Verifies that this type references a valid type parameter, relative to the given `op`.
    ::mlir::LogicalResult verifySymbol(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);
  }];
}

def Root : ZMLType<"Root", "root", [ComponentLike]> {
  let summary = "root type";
  let description = "type used to represent the 'Component' builtin";
  let extraClassDeclaration = [{
    ::mlir::FlatSymbolRefAttr getName() {
      return ::mlir::FlatSymbolRefAttr::get(
        ::mlir::StringAttr::get(getContext(), "Component")
      );
    }
    ::zml::ComponentLike getSuperType() { return nullptr;}
    ::mlir::ArrayRef<::mlir::Attribute> getParams() { return {};}
    bool getBuiltin() { return true;}

      ::zml::ComponentInterface getDefinition(
          ::mlir::SymbolTableCollection & , ::mlir::Operation * ) { return nullptr; }
    bool isRoot() { return true;}
    bool isConcreteArray() { return  false;
}
bool isArray() { return false; }
// ::mlir::FailureOr<::mlir::Type> getFirstMatchingSuperType(
//     llvm::function_ref<bool(::mlir::Type)>);
::mlir::FailureOr<::mlir::Type> getArrayInnerType() {
  return ::mlir::failure();
}
::mlir::FailureOr<::mlir::Attribute> getArraySize() {
  return ::mlir::failure();
}
::zml::ComponentLike getSuperTypeAsComp() { return nullptr; }
bool subtypeOf(::mlir::Type t) { return ::mlir::isa<::zml::RootType>(t); }
}];
}

def ExtVal : ZMLType<"ExtVal", "extval", [ComponentLike]> {
  let summary = "field polynomial type";
  let description = "type used to represent the 'ExtVal' builtin";
  let extraClassDeclaration = [{
    ::mlir::FlatSymbolRefAttr getName() {
      return ::mlir::FlatSymbolRefAttr::get(
        ::mlir::StringAttr::get(getContext(), "ExtVal")
      );
    }
    ::zml::ComponentLike getSuperType() { return ::zml::RootType::get(getContext()); }
    ::mlir::ArrayRef<::mlir::Attribute> getParams() { return {}; }
    bool getBuiltin() { return true; }

      ::zml::ComponentInterface getDefinition(
          ::mlir::SymbolTableCollection & , ::mlir::Operation * ) { return nullptr;}
    bool isRoot() { return false; }
    bool isConcreteArray() { return false;}
    bool isArray() { return false;}
    // ::mlir::FailureOr<::mlir::Type> getFirstMatchingSuperType(llvm::function_ref<bool(::mlir::Type)>);
    ::mlir::FailureOr<::mlir::Type> getArrayInnerType() { return ::mlir::failure(); }
    ::mlir::FailureOr<::mlir::Attribute> getArraySize() { return ::mlir::failure(); }
  // ::zml::ComponentLike getSuperTypeAsComp() { return ::zml::RootType::get(getContext()); }
  bool subtypeOf(::mlir::Type t) { return ::mlir::isa<::zml::ExtValType, ::zml::RootType>(t); }
}];
}

def ComplexComponent : ZMLType<"ComplexComponent", "comp", [ComponentLike]> {
  let summary = "component type";
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$name,
      OptionalParameter<"::zml::ComponentLike">:$superType,
      OptionalArrayRefParameter<"::mlir::Attribute">:$params, "bool":$builtin);
  let assemblyFormat = [{
   `<` $name $builtin (`<` $params^ `>`)?(`:` $superType^)?  `>` }];

  let extraClassDeclaration = [{

    /// Gets the `component` op that defin es this  componnet.Provided `op` is
      /// used as a starting point for the  lookup.Should not be ass ume d to
      /// be non-`null` as we don't verify all types during verification.
      ::zml::ComponentInterface getDefinition(
          ::mlir::SymbolTableCollection & symbolTable, ::mlir::Operation * op);

  // Verifies that this type references a valid component, relative to the given
  // `op`.
  ::mlir::LogicalResult verifySymbol(
      ::mlir::SymbolTableCollection & symbolTable, ::mlir::Operation * op);

  /// Returns wether the component associated to this type is the root
  /// 'Component'
  bool isRoot() const { return false; }
  bool isConcreteArray() const { return false; }

  ::mlir::FailureOr<mlir::Type> getArrayInnerType() const;
  ::mlir::FailureOr<mlir::Attribute> getArraySize() const;
  // ComponentLike getSuperTypeAsComp() const;
  bool subtypeOf(mlir::Type) const;
}];

  let builders = [
    TypeBuilder<(ins "::mlir::StringRef":$name, 
                                  "::zml::ComponentLike":$superType,
                                  "::llvm::ArrayRef<::mlir::Attribute>":$params,
                                  CArg<"bool", "false">:$builtin), [{
    return Base::get($_ctxt, mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get($_ctxt, name)), superType, params, builtin);
  }]>,
    TypeBuilderWithInferredContext<(ins "::mlir::StringAttr":$name, 
                                  "::zml::ComponentLike":$superType,
                                  "::llvm::ArrayRef<::mlir::Attribute>":$params,
                                  CArg<"bool", "false">:$builtin), [{
      return Base::get(name.getContext(), mlir::FlatSymbolRefAttr::get(name), superType, params, builtin);
  }]>,
    TypeBuilderWithInferredContext<(ins "::mlir::FlatSymbolRefAttr":$name, 
                                  "::zml::ComponentLike":$superType,
                                  "::llvm::ArrayRef<::mlir::Attribute>":$params,
                                  CArg<"bool", "false">:$builtin), [{
      return Base::get(name.getContext(), name, superType, params, builtin);
  }]>,
  TypeBuilder<(ins "::mlir::StringRef":$name, 
                                  "::zml::ComponentLike":$superType,
                                  CArg<"bool", "false">:$builtin), [{
    return Base::get($_ctxt, mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get($_ctxt, name)), superType, std::nullopt, builtin);
  }]>,
    TypeBuilderWithInferredContext<(ins "::mlir::StringAttr":$name, 
                                  "::zml::ComponentLike":$superType,
                                  CArg<"bool", "false">:$builtin), [{
      return Base::get(name.getContext(), mlir::FlatSymbolRefAttr::get(name), superType, std::nullopt, builtin);
  }]>,
    TypeBuilderWithInferredContext<(ins "::mlir::FlatSymbolRefAttr":$name, 
                                  "::zml::ComponentLike":$superType,
                                  CArg<"bool", "false">:$builtin), [{
      return Base::get(name.getContext(), name, superType, std::nullopt, builtin);
  }]>,



];

      let genVerifyDecl = 1;
}

def VarArgs : ZMLType<"VarArgs", "vargs"> {
  let summary = "variable number of arguments";

  let parameters = (ins "::mlir::Type" : $inner);
  let assemblyFormat = "`<` $inner `>`";
}

def AnyZMLType : AnyTypeOf<[TypeVar, ComponentLike, VarArgs, Index],
                           "type that can participate in ZML components">;

class TypeVarOr<Type type> : AnyTypeOf<[type, TypeVar]>;

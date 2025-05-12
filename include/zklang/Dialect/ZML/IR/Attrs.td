//===- Attrs.td - ZML attributes definitions ---------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes attributes to define compile time constant expressions.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "Dialect.td"

class ZMLAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ZMLDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Variadic : ZMLAttr<"Variadic", "variadic"> {} // XXX: Remove me??

def ConstExpr : ZMLAttr<"ConstExpr", "cexpr"> {
  let summary = "constant expression";
  let description = [{
    Represents a constant expression that is affine representable. It is meant for translating affine maps 
    in component params from the caller's scope. Includes a list of formals that maps what other parameters from the 
    component are meant to be used as symbol arguments in the affine map.

    For example `component Bar<N: Val, M: Val>(foo: Foo<N + M>)` will generate the following attribute:

    ``` 
    cexpr< ()[s0, s1] -> (s0 + s1), formals = 0, 1 >
    ```
  }];
  let parameters = (ins "mlir::AffineMap":$map,
      ArrayRefParameter<"uint64_t", "">:$formals);

  let builders = [AttrBuilderWithInferredContext<
      (ins "mlir::AffineMap":$map, "mlir::ArrayRef<uint64_t>":$formals), [{
      return $_get(map.getContext(), map, formals);
    }]>];

  let assemblyFormat = "`<` $map `,` `formals` `=` $formals `>`";
}

def LiftedExpr : ZMLAttr<"LiftedExpr", "lexpr"> {
  let summary = "constant expression associated to a name";
  let description = [{
    Represents a constant expression that is affine representable that has been associated with a parameter. It is meant for translating affine maps 
    in component params from the caller's scope. Includes a list of formals that maps what other parameters from the 
    component are meant to be used as symbol arguments in the affine map. 

    For example in `component Bar<N: Val, M: Val>(foo: Foo<N + M>)` the sum will get lifted in to
      a parameter in component Bar,
      thus generating the following attribute for the lifted attribute :

    ``` lexpr < @"Aff$0" : cexpr < ()[s0, s1]->(s0 + s1), formals = 0, 1 >>
    ```
}];

let parameters = (ins "::mlir::FlatSymbolRefAttr" : $symbol, ConstExpr : $expr);

  let builders = [AttrBuilderWithInferredContext<
      (ins "mlir::FlatSymbolRefAttr":$symbol, "zml::ConstExprAttr":$cexpr), [{
      return $_get(symbol.getContext(), symbol, cexpr);
    }]>];
  let assemblyFormat = "`<` $symbol `:` $expr `>`";
}

def GenericParam : ZMLAttr<"GenericParam", "generic"> {
  let summary = "generic parameter";
  let description = [{
    A type binding for a generic parameter. Has an associated name 
    and inner type binding. It can also be marked as injected for 
    parameters that have been created as part of the type checking process.
  }];

  let parameters = (ins "mlir::StringAttr":$name, "zml::FixedTypeBindingAttr":$type, "bool":$injected);

  let assemblyFormat = "`<` struct(params) `>`";
}

def ComponentMember : ZMLAttr<"ComponentMember", "member"> {
  let summary = "component member";
  let description = [{
    Holds information of a member of a component; name, type, and visibility.
  }];

  let parameters = (ins "mlir::StringAttr":$name, "zml::FixedTypeBindingAttr":$type, "bool":$isPublic);

  let assemblyFormat = "`<` struct(params) `>`";
}

def FixedTypeBinding : ZMLAttr<"FixedTypeBinding", "binding", [TypedAttrInterface]> {
  let summary = "type binding";
  let description = [{
    The solidified result of type checking a ZHL operation.
    The `typecheck-zhl` pass will inject this attribute into the ZHL operations
    defined in the circuit. The subsequent passes depend on this information to 
    know what final LLZK type the operation is going to have.
  }];

  let parameters = (ins 
    AttributeSelfTypeParameter<"">:$type,
  "mlir::StringAttr":$name,
  ArrayRefParameter<"zml::GenericParamAttr">:$genericParams,
  ArrayRefParameter<"zml::ComponentMemberAttr">:$members,
  "mlir::FunctionType":$ctorType,
  ArrayRefParameter<"mlir::LocationAttr">:$paramLocs,
  OptionalParameter<"zml::FixedTypeBindingAttr">:$superType,
  OptionalParameter<"mlir::Attribute">:$constExpr,
    "bool":$builtin
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let builders = [
    AttrBuilder<(ins "const zhl::TypeBinding &":$typeBinding, "const zhl::TypeBindings &":$bindings)>
  ];
}

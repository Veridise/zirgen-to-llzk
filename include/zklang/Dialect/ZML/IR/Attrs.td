// Copyright 2024 Veridise, Inc.

include "mlir/IR/AttrTypeBase.td"
include "Dialect.td"

class ZMLAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ZMLDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Variadic : ZMLAttr<"Variadic", "variadic"> {}

def ConstExpr : ZMLAttr<"ConstExpr", "cexpr"> {
  let summary = "constant expression";
  let description = [{
    Represents a constant expression that is affine representable. It is meant for translating affine maps 
    in component params from the caller's scope. Includes a list of formals that maps what other parameters from the 
    component are meant to be used as symbol arguments in the affine map.
  }];
  let parameters = (ins "mlir::AffineMap":$map,
      ArrayRefParameter<"uint64_t", "">:$formals);

  let builders = [AttrBuilderWithInferredContext<
      (ins "mlir::AffineMap":$map, "mlir::ArrayRef<uint64_t>":$formals), [{
      return $_get(map.getContext(), map, formals);
    }]>];

  let assemblyFormat = "`<` $map `,` `formals` `=` $formals `>`";
}

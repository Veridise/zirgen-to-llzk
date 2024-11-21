// Copyright 2024 Veridise, Inc.

include "mlir/IR/AttrTypeBase.td"
include "ZirToZkir/Dialect/ZMIR/IR/Dialect.td"
include "ZirToZkir/Dialect/ZMIR/IR/Attrs.td"
include "mlir/IR/SymbolInterfaces.td"

class ZmirType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ZmirDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Val : ZmirType<"Val", "val"> { let summary = "value type"; }

def TypeVar : ZmirType<"TypeVar", "tvar"> {
  let summary = "parametric type linked to a symbol";

  let parameters = (ins "::mlir::FlatSymbolRefAttr":$name);
  let assemblyFormat = [{ `<` $name `>` }];

  let extraClassDeclaration = [{
    // Verifies that this type references a valid type parameter, relative to the given `op`.
    ::mlir::LogicalResult verifySymbol(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);
  }];
}

def String : ZmirType<"String", "string"> { let summary = "string type"; }

def Union : ZmirType<"Union", "union"> { let summary = "union type"; }

def Component : ZmirType<"Component", "component"> {
  let summary = "component type";
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$name,
      OptionalParameter<"::mlir::Type">:$super,
      OptionalArrayRefParameter<"::mlir::Attribute">:$type_params,
      OptionalArrayRefParameter<"::mlir::Attribute">:$const_params);
  let assemblyFormat =
      [{ `<` $name (`<` $type_params^ `>`)? (`(` $const_params^ `)`)? `>` ( `:` $super^ )? }];

  let extraClassDeclaration = [{

    ::mlir::Type getSuperType();
    /// Gets the `component` op that defines this componnet. Provided `op` is
    /// used as a starting point for the lookup. Should not be assumed to
    /// be non-`null` as we don't verify all types during verification.
    /*ComponentOp getDefinition(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);*/
  
    // Verifies that this type references a valid component, relative to the given `op`.
    ::mlir::LogicalResult verifySymbol(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);
  }];

  let builders =
      [TypeBuilder<(ins "::llvm::StringRef":$name, "::mlir::Type":$super,
                       "::llvm::ArrayRef<::mlir::Attribute>":$type_params,
                       "::llvm::ArrayRef<::mlir::Attribute>":$const_params),
                   [{
      return Base::get($_ctxt, mlir::SymbolRefAttr::get(mlir::StringAttr::get($_ctxt, name)), super, type_params, const_params);
}]>,
       TypeBuilder<(ins "::llvm::StringRef":$name,
                       "::llvm::ArrayRef<::mlir::Attribute>":$type_params,
                       "::llvm::ArrayRef<::mlir::Attribute>":$const_params),
                   [{
      return Base::get($_ctxt, mlir::SymbolRefAttr::get(mlir::StringAttr::get($_ctxt, name)), nullptr, type_params, const_params);
}]>,
       TypeBuilder<(ins "::llvm::StringRef":$name, "::mlir::Type":$super), [{
      return Base::get($_ctxt, mlir::SymbolRefAttr::get(mlir::StringAttr::get($_ctxt, name)), super, std::nullopt, std::nullopt);
    }]>,

       TypeBuilder<(ins "::llvm::StringRef":$name), [{
      return Base::get($_ctxt, mlir::SymbolRefAttr::get(mlir::StringAttr::get($_ctxt, name)), nullptr, std::nullopt, std::nullopt);
    }]>];

  let genVerifyDecl = 1;
}

def Array : ZmirType<"Array", "array", []> {
  let summary = "array type";
  let parameters = (ins "::mlir::Type":$inner_type, "::mlir::Attribute":$size);
  let assemblyFormat = [{ `<` $inner_type `,` $size `>`}];
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    uint64_t getSizeInt();
  }];
}

def Pending : ZmirType<"Pending", "pending"> {}

def VarArgs : ZmirType<"VarArgs", "vargs"> {
  let summary = "represents a variable number of arguments for a function";

  let parameters = (ins "::mlir::Type":$inner);
  let assemblyFormat = "`<` $inner `>`";
}

def AnyZmirType
    : AnyTypeOf<[Val, TypeVar, String, Union, Component, Array, Pending],
                "A type that can participate in ZMIR components">;

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
      OptionalArrayRefParameter<"::mlir::FlatSymbolRefAttr">:$type_params,
      OptionalArrayRefParameter<"::mlir::FlatSymbolRefAttr">:$const_params);
  let assemblyFormat =
      [{ `<` $name (`<` $type_params^ `>`)? (`(` $const_params^ `)`)? `>` }];

  let extraClassDeclaration = [{

    /// Gets the `component` op that defines this componnet. Provided `op` is
    /// used as a starting point for the lookup. Should not be assumed to
    /// be non-`null` as we don't verify all types during verification.
    ComponentOp getDefinition(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);
  
    // Verifies that this type references a valid component, relative to the given `op`.
    ::mlir::LogicalResult verifySymbol(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);
  }];

  let builders =
      [TypeBuilder<
           (ins "::llvm::StringRef":$name,
               "::llvm::ArrayRef<::mlir::FlatSymbolRefAttr>":$type_params,
               "::llvm::ArrayRef<::mlir::FlatSymbolRefAttr>":$const_params),
           [{
      return Base::get($_ctxt, mlir::SymbolRefAttr::get(mlir::StringAttr::get($_ctxt, name)), type_params, const_params);
}]>,
       TypeBuilder<(ins "::llvm::StringRef":$name), [{
      return Base::get($_ctxt, mlir::SymbolRefAttr::get(mlir::StringAttr::get($_ctxt, name)), std::nullopt, std::nullopt);
    }]>];
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

def AnyZmirType : AnyTypeOf<[Val, TypeVar, String, Union, Component, Array],
                            "A type that can participate in ZMIR components">;

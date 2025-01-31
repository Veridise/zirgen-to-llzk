// Copyright 2024 Veridise, Inc.

include "mlir/IR/AttrTypeBase.td"
include "Dialect.td"
include "Attrs.td"
include "TypeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

class ZmirType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ZmirDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Val : ZmirType<"Val", "val"> { let summary = "value type"; }

def TypeVar : ZmirType<"TypeVar", "tvar"> {
  let summary = "parametric type";

  let parameters = (ins "::mlir::FlatSymbolRefAttr":$name);
  let assemblyFormat = [{ `<` $name `>` }];

  let extraClassDeclaration = [{
    // Verifies that this type references a valid type parameter, relative to the given `op`.
    ::mlir::LogicalResult verifySymbol(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);
  }];
}

def String : ZmirType<"String", "string"> { let summary = "string type"; }

def Union : ZmirType<"Union", "union"> { let summary = "union type"; }

def Component : ZmirType<"Component", "comp"> {
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
    ::zkc::Zmir::ComponentInterface getDefinition(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);
  
    // Verifies that this type references a valid component, relative to the given `op`.
    ::mlir::LogicalResult verifySymbol(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);

    /// Returns wether the component associated to this type is the root 'Component'
    bool isRoot() const { return getName().getValue() == "Component"; }

    static ComponentType getRoot(mlir::MLIRContext *ctx) { return ComponentType::get(ctx); }
    static ComponentType Component(mlir::MLIRContext *ctx) { return ComponentType::get(ctx); }
    static ComponentType Val(mlir::MLIRContext *ctx) { return ComponentType::get(ctx, "Val", Component(ctx), true); }
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

def BoundedArray : ZmirType<"BoundedArray", "array", [Array]> {
  let summary = "array type";
  let parameters = (ins "::mlir::Type":$inner_type, "::mlir::Attribute":$size);
  let assemblyFormat = [{ `<` $inner_type `,` $size `>`}];
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    int64_t getSizeInt();
  }];
}

def UnboundedArray : ZmirType<"UnboundedArray", "uarray", [Array]> {
  let summary = "unbounded array type";
  let parameters = (ins "::mlir::Type":$inner_type);
  let assemblyFormat = "`<` $inner_type `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    int64_t getSizeInt() { return -1; }
    mlir::Attribute getSize();
  }];
}

def Pending : ZmirType<"Pending", "pending"> { let summary = "pending type"; }

def VarArgs : ZmirType<"VarArgs", "vargs"> {
  let summary = "variable number of arguments";

  let parameters = (ins "::mlir::Type":$inner);
  let assemblyFormat = "`<` $inner `>`";
}

def AnyNonPendingZmirType
    : AnyTypeOf<[Val, TypeVar, String, Union, Component, Array, VarArgs],
                "type that can participate in ZMIR components">;
def AnyZmirType
    : AnyTypeOf<
          [AnyNonPendingZmirType, Pending],
          "potentially unknown type that can participate in ZMIR components">;

class PendingOf<Type type> : AnyTypeOf<[type, Pending]>;
class TypeVarOr<Type type> : AnyTypeOf<[type, TypeVar]>;

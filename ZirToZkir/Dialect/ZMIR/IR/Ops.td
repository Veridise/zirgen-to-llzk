// Copyright 2024 Veridise, Inc.


include "mlir/IR/OpBase.td"
include "ZirToZkir/Dialect/ZMIR/IR/Dialect.td"
include "ZirToZkir/Dialect/ZMIR/IR/Types.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/CommonAttrConstraints.td"

def ReturnsExpr : DeclareOpInterfaceMethods<InferTypeOpInterface>;

class ZmirOp<string mnemonic, list<Trait> traits = []>
    : Op<ZmirDialect, mnemonic, traits> {
}

def ComponentOp : ZmirOp<"component", [
  HasParent<"::mlir::ModuleOp">,
  Symbol,
  SymbolTable,
  IsolatedFromAbove,
  GraphRegionNoTerminator,
  OpAsmOpInterface
]> {
  let summary = "component declaration";
  let arguments = (ins 
    SymbolNameAttr:$sym_name, 
    UnitAttr:$function, 
    UnitAttr:$generic, 
    OptionalAttr<FlatSymbolRefArrayAttr>:$const_params,
    OptionalAttr<FlatSymbolRefArrayAttr>:$type_params
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];
}

def BodyOp : ZmirOp<"body", [
  HasParent<"::zkc::Zmir::ComponentOp">, FunctionOpInterface
]> {
  let summary = "body of a component";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  // let regions = (region AnyRegion:$body);
  let regions = (region SizedRegion<1>:$body);
  // let assemblyFormat = [{
  //   attr-dict-with-keyword $body
  // }];
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return &getBody(); }

    /// Returns the argument types of this function.
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
}

def ReturnOp: ZmirOp<"return", [
  Terminator,
  HasParent<"::zkc::Zmir::BodyOp">
]> {
  let summary = "return";
  let description = "Returns a value from the body of a component";
  let arguments = (ins AnyZmirType:$value);
  let assemblyFormat = [{
    $value attr-dict `:` type($value)
  }];
}

def ConstructOp: ZmirOp<"construct", [
  HasParent<"::zkc::Zmir::BodyOp">,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "construct component";
  let description = "Constructs and executes new component";

  let arguments = (ins SymbolRefAttr:$component, Variadic<AnyZmirType>:$operands);
  let results = (outs AnyZmirType);

  let assemblyFormat = [{
    $component `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

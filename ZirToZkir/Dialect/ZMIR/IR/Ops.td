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
include "mlir/Interfaces/SideEffectInterfaces.td"

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
    UnitAttr:$builtin, 
    OptionalAttr<FlatSymbolRefArrayAttr>:$const_params,
    OptionalAttr<FlatSymbolRefArrayAttr>:$type_params
  );
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "::mlir::StringRef":$name), [{
      $_state.getOrAddProperties<Properties>().sym_name = $_builder.getStringAttr(name);
      $_state.addRegion();
    }]>,
    OpBuilder<(ins "::mlir::StringRef":$name, "::zkc::Zmir::IsBuiltIn":$tag), [{
      $_state.getOrAddProperties<Properties>().sym_name = $_builder.getStringAttr(name);
      $_state.getOrAddProperties<Properties>().builtin = $_builder.getUnitAttr();
      $_state.addRegion();
    }]>
  ];

  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];
}

def FieldDefOp : ZmirOp<"field", [
  HasParent<"::zkc::Zmir::ComponentOp">, Symbol
]> {
  let summary = "component field definition";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyZmirType>:$type);

  let assemblyFormat = [{
    $sym_name `:` $type attr-dict
  }];
}

def ReadFieldOp : ZmirOp<"read_field", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "reads the value of a field";
  let description = [{
    Reads the value of a field in the current component.
    In ZIR you can read fields of other components but
    how to support that is TBD.
  }];

  let arguments = (ins FlatSymbolRefAttr:$field_name);
  let results = (outs AnyZmirType:$val);

  let assemblyFormat = [{
    $field_name attr-dict `:` type($val)
  }];
}

def WriteFieldOp : ZmirOp<"write_field", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "writes a value into a field";
  let description = [{
    Writes a value into a field in the current component.
  }];

  let arguments = (ins FlatSymbolRefAttr:$field_name, AnyZmirType:$val);

  let assemblyFormat = [{
    $field_name `=` $val attr-dict `:` type($val)
  }];
}

def GetSelfOp: ZmirOp<"self"> {
  let summary = "get current component";
  let description = "Returns the current component";

  let results = (outs Component);

  let assemblyFormat = [{
     attr-dict `:` type(results)
  }];
}

///////////////////// Builtin Ops /////////////////////

def InRangeOp : ZmirOp<"in_range", [Pure, SameOperandsAndResultType]> {
  let summary = "checks if value is within range";
  let arguments = (ins Val:$low, Val:$mid, Val:$high);
  let results = (outs Val:$out);
  let assemblyFormat = [{ $low ` ` `<` `=` $mid ` ` `<` ` ` $high `:` type($out) attr-dict }];
}

class UnaryOp<string mnemonic, list<Trait> traits = []>
    : ZmirOp<mnemonic, !listconcat([Pure/*, InferTypeOpAdaptor*/], traits)> {
  let arguments = (ins Val:$in);
  let results = (outs Val:$out);
  let assemblyFormat = [{ $in `:` type($in) attr-dict }];
  // let hasFolder = 1;
}

def IsZeroOp : UnaryOp<"isz"> {}
def NegOp : UnaryOp<"neg"> {}
def InvOp : UnaryOp<"inv"> {}

class BinaryOp<string mnemonic, list<Trait> traits = []>
    : ZmirOp<mnemonic, !listconcat([Pure/*, InferTypeOpAdaptor*/], traits)> {
  let arguments = (ins Val:$lhs, Val:$rhs);
  let results = (outs Val:$out);
  let assemblyFormat = [{ $lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict }];
  // let hasFolder = 1;
}

def AddOp : BinaryOp<"add", [Commutative]> {}
def SubOp : BinaryOp<"sub"> {}
def MulOp : BinaryOp<"mul", [Commutative]> {}
def BitAndOp : BinaryOp<"bit_and"> {}

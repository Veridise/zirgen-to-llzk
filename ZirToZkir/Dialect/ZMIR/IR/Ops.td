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
    : Op<ZmirDialect, mnemonic, traits> {}

def ComponentOp
    : ZmirOp<"component", [HasParent<"::mlir::ModuleOp">, Symbol, SymbolTable,
                           IsolatedFromAbove, GraphRegionNoTerminator,
                           OpAsmOpInterface]> {
  let summary = "component declaration";
  let arguments = (ins SymbolNameAttr:$sym_name, UnitAttr:$function,
      UnitAttr:$generic, UnitAttr:$builtin,
      OptionalAttr<FlatSymbolRefArrayAttr>:$const_params,
      OptionalAttr<FlatSymbolRefArrayAttr>:$type_params);
  let regions = (region SizedRegion<1>:$body);

  let builders =
      [OpBuilder<(ins "::mlir::StringRef":$name), [{
      $_state.getOrAddProperties<Properties>().sym_name = $_builder.getStringAttr(name);
      $_state.addRegion();
    }]>,
       OpBuilder<(ins "::mlir::StringRef":$name, "::zkc::Zmir::IsBuiltIn":$tag),
                 [{
      $_state.getOrAddProperties<Properties>().sym_name = $_builder.getStringAttr(name);
      $_state.getOrAddProperties<Properties>().builtin = $_builder.getUnitAttr();
      $_state.addRegion();
    }]>,

       OpBuilder<(ins "::mlir::StringRef":$name,
                     "::mlir::ArrayRef<mlir::Attribute>":$type_params,
                     "::mlir::ArrayRef<mlir::Attribute>":$const_params,
                     "::zkc::Zmir::IsBuiltIn":$tag),
                 [{
      $_state.getOrAddProperties<Properties>().sym_name = $_builder.getStringAttr(name);
      $_state.getOrAddProperties<Properties>().builtin = $_builder.getUnitAttr();
      if (type_params.size() + const_params.size() > 1) {
        $_state.getOrAddProperties<Properties>().generic = $_builder.getUnitAttr();
        $_state.getOrAddProperties<Properties>().type_params = $_builder.getArrayAttr(type_params);
        $_state.getOrAddProperties<Properties>().const_params = $_builder.getArrayAttr(const_params);
      }

      $_state.addRegion();
    }]>];

  let assemblyFormat = [{
    $sym_name (`<` $type_params^ `>`)? (`(` $const_params^ `)`)? attr-dict-with-keyword $body
  }];
}

def FieldDefOp
    : ZmirOp<"field", [HasParent<"::zkc::Zmir::ComponentOp">, Symbol]> {
  let summary = "component field definition";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyZmirType>:$type);

  let assemblyFormat = [{
    $sym_name `:` $type attr-dict
  }];
}

def ReadFieldOp
    : ZmirOp<"read_field", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "reads the value of a field";
  let description = [{
    Reads the value of a field in a component.
  }];

  let arguments = (ins Component:$component, FlatSymbolRefAttr:$field_name);
  let results = (outs AnyZmirType:$val);

  let assemblyFormat = [{
    $component `[` $field_name `]` attr-dict `:` type($component) `,` type($val) 
  }];
}

def WriteFieldOp
    : ZmirOp<
          "write_field", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "writes a value into a field";
  let description = [{
    Writes a value into a field in a component.
  }];

  let arguments = (ins Component:$component, FlatSymbolRefAttr:$field_name,
      AnyZmirType:$val);

  let assemblyFormat = [{
    $component `[` $field_name `]` `=` $val attr-dict `:` type($component) `,` type($val) 
  }];
}

def GetSelfOp : ZmirOp<"self"> {
  let summary = "get current component";
  let description = "Returns the current component";

  let results = (outs Component);

  let assemblyFormat = [{
     attr-dict `:` type(results)
  }];
}

def GlobalDefOp : ZmirOp<"global", [HasParent<"::mlir::ModuleOp">]> {
  let summary = "defines a global component";
  let description = "Defines a global component and binds it to a symbol.";

  let arguments = (ins SymbolNameAttr:$sym_name, AnyZmirType:$value,
      TypeAttrOf<AnyZmirType>:$type);

  let assemblyFormat = [{
    $sym_name `=` $value `:` $type `,` type($value) attr-dict
  }];
}

def GetGlobalOp
    : ZmirOp<"get_global", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "loads a global variable";

  let arguments = (ins SymbolNameAttr:$sym_name);

  let assemblyFormat = [{
    $sym_name `:` attr-dict
  }];
}

def ConstrainOp : ZmirOp<"constrain"> {
  let summary = "emits a constraint";

  let arguments = (ins AnyZmirType:$lhs, AnyZmirType:$rhs);

  let assemblyFormat = [{
  $lhs `=` $rhs attr-dict `:` type($lhs) `,` type($rhs)
  }];
}

def NewArrayOp
    : ZmirOp<"new_array",
             [TypesMatchWith<
                 "operand types match result type", "result", "elements",
                 "::llvm::SmallVector<::mlir::Type, 2>("
                 "::llvm::cast<::zkc::Zmir::ArrayType>($_self).getSizeInt(), "
                 "::llvm::cast<::zkc::Zmir::ArrayType>($_self).getInnerType())">

]> {
  let summary = "creates a new array with data";

  let arguments = (ins Variadic<AnyZmirType>:$elements);
  let results = (outs Array:$result);

  let assemblyFormat = "$elements attr-dict `:` type($result)";
}

def AllocArrayOp : ZmirOp<"alloc_array"> {
  let summary = "creates a new empty array";

  let results = (outs Array:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def ReadArrayOp : ZmirOp<"read_array"> {
  let summary = "read data from an array";

  let arguments = (ins Array:$lvalue, Variadic<Val>:$indices);
  let results = (outs AnyZmirType:$result);

  let assemblyFormat = [{
    $lvalue `[` $indices `]` attr-dict `:` type($lvalue) `,` type($result)
  }];
}

def LitValOp : ZmirOp<"literal"> {
  let summary = "a literal value";
  let arguments = (ins UI64Attr:$value);
  let results = (outs Val);

  let assemblyFormat = [{
  $value `:` attr-dict
  }];
}

def LitStrOp : ZmirOp<"string"> {
  let summary = "a literal value";
  let arguments = (ins StrAttr:$value);
  let results = (outs String);

  let assemblyFormat = [{
  $value `:` attr-dict
  }];
}

///////////////////// Builtin Ops /////////////////////

def InRangeOp : ZmirOp<"in_range", [Pure, SameOperandsAndResultType]> {
  let summary = "checks if value is within range";
  let arguments = (ins Val:$low, Val:$mid, Val:$high);
  let results = (outs Val:$out);
  let assemblyFormat =
      [{ $low ` ` `<` `=` $mid ` ` `<` ` ` $high `:` type($out) attr-dict }];
}

class UnaryOp<string mnemonic, list<Trait> traits = []>
    : ZmirOp<mnemonic, !listconcat([Pure /*, InferTypeOpAdaptor*/], traits)> {
  let arguments = (ins Val:$in);
  let results = (outs Val:$out);
  let assemblyFormat = [{ $in `:` type($in) attr-dict }];
  // let hasFolder = 1;
}

def IsZeroOp : UnaryOp<"isz"> {}
def NegOp : UnaryOp<"neg"> {}
def InvOp : UnaryOp<"inv"> {}

class BinaryOp<string mnemonic, list<Trait> traits = []>
    : ZmirOp<mnemonic, !listconcat([Pure /*, InferTypeOpAdaptor*/], traits)> {
  let arguments = (ins Val:$lhs, Val:$rhs);
  let results = (outs Val:$out);
  let assemblyFormat =
      [{ $lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict }];
  // let hasFolder = 1;
}

def AddOp : BinaryOp<"add", [Commutative]> {}
def SubOp : BinaryOp<"sub"> {}
def MulOp : BinaryOp<"mul", [Commutative]> {}
def BitAndOp : BinaryOp<"bit_and"> {}

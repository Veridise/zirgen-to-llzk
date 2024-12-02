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
include "ZirToZkir/Dialect/ZMIR/IR/OpInterfaces.td"

def ReturnsExpr : DeclareOpInterfaceMethods<InferTypeOpInterface>;

class ZmirOp<string mnemonic, list<Trait> traits = []>
    : Op<ZmirDialect, mnemonic, traits> {}

class ComponentOpBase<string prefix, string bodyFuncName,
                      string constrainFuncName>
    : ZmirOp<prefix#"component", [HasParent<"::mlir::ModuleOp">, Symbol,
                                  SymbolTable, IsolatedFromAbove,
                                  GraphRegionNoTerminator, OpAsmOpInterface,
                                  ComponentInterface]> {
  let summary = prefix#"component declaration";
  let arguments = (ins SymbolNameAttr:$sym_name, UnitAttr:$function,
      UnitAttr:$generic, UnitAttr:$builtin,
      OptionalAttr<FlatSymbolRefArrayAttr>:$const_params,
      OptionalAttr<FlatSymbolRefArrayAttr>:$type_params);
  let regions = (region SizedRegion<1>:$body);

  let builders = [OpBuilder<(ins "::mlir::StringRef":$name,
                      "::mlir::ArrayRef<mlir::NamedAttribute>":$attrs)>,
                  OpBuilder<(ins "::mlir::StringRef":$name,
                      "::zkc::Zmir::IsBuiltIn":$tag)>,
                  OpBuilder<(ins "::mlir::StringRef":$name,
                      "::mlir::ArrayRef<mlir::StringRef>":$type_params,
                      "::mlir::ArrayRef<mlir::StringRef>":$const_params,
                      "::zkc::Zmir::IsBuiltIn":$tag)>,
                  OpBuilder<(ins "::mlir::StringRef":$name,
                      "::mlir::ArrayRef<mlir::StringRef>":$type_params,
                      "::mlir::ArrayRef<mlir::StringRef>":$const_params,
                      "::mlir::ArrayRef<mlir::NamedAttribute>":$attrs)>];

  let extraClassDeclaration = [{
    /// Returns the type of the component.
    ::mlir::Type getType();
    ::mlir::FailureOr<::mlir::Type> getSuperType();
    ::mlir::FailureOr<::mlir::Type> lookupFieldType(mlir::FlatSymbolRefAttr);

    /// Returns the name of the function that represents the body of a component.
    std::string_view getBodyFuncName() { return "}]#bodyFuncName#[{"; }
    std::string_view getConstrainFuncName() { return "}]#constrainFuncName#[{"; }

    bool hasUnifiedBody() { return getBodyFuncName() == getConstrainFuncName(); }

    mlir::FlatSymbolRefAttr getSuperFieldName() { return mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(getContext(), "$super")); }

    bool isRoot() { return getSymName() == "Component"; }
  }];

  let assemblyFormat = [{
    $sym_name (`<` $type_params^ `>`)? (`(` $const_params^ `)`)? attr-dict-with-keyword $body
  }];
}

def ComponentOp : ComponentOpBase<"", "body", "body"> {}

def SplitComponentOp : ComponentOpBase<"split_", "compute", "constrain"> {}

//===----------------------------------------------------------------------===//
//  ConstructorRefOp
//===----------------------------------------------------------------------===//

def ConstructorRefOp : ZmirOp<"constructor", [Pure]> {
  let summary = "reference to a component constructor";
  let description = [{
    Returns a function value for the constructor function of a component.

    The constructor is referenced by the name of the component, for example:

    ```
    %0 = zmir.constructor @ExampleComponent : (!zmir.val) -> !zmir.component<@ExampleComponent>
    %1 = func.call_indirect %0(%2) : !zmir.component<@ExampleComponent>
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$component, UnitAttr:$builtin);
  let results = (outs FunctionType);
  let assemblyFormat = "attr-dict $component `:` type(results)";

  let builders = [OpBuilder<(ins "ComponentInterface":$comp)>];

  let extraClassDeclaration = [{
    /// Returns true if can be built with the given value
    /// and result type.
    static bool isBuildableWith(mlir::Attribute value, mlir::Type type);
  }];

  let hasVerifier = 1;
}

def FieldDefOp : ZmirOp<"field", [HasParent<"::zkc::Zmir::ComponentOp, "
                                            "::zkc::Zmir::SplitComponentOp">,
                                  Symbol]> {
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

  let arguments = (ins PendingOf<Component>:$component,
      FlatSymbolRefAttr:$field_name);
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

def ReadSuperTransOp : ZmirOp<"read_super_tr"> {
  let summary =
      "transitively reads the super field until a primitive value is reached";

  let arguments = (ins Component:$component);
  let results = (outs AnyNonPendingZmirType:$val);

  let assemblyFormat =
      "$component attr-dict `:` type($component) `,` type($val)";
}

def NopOp : ZmirOp<"nop", [Pure]> {

  let arguments = (ins Variadic<AnyZmirType>:$ins);
  let results = (outs Variadic<AnyZmirType>:$outs);

  let assemblyFormat = "$ins `:` type($ins) `->` type($outs) attr-dict";
}

def GetSelfOp : ZmirOp<"self", [Pure]> {
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

  let arguments = (ins Val:$lhs, Val:$rhs);

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

def VarArgsTypeCheckPred : TypesMatchWith<"inner type matches input operands",
                                          "result", "elements", [{
  ::llvm::cast<::zkc::Zmir::VarArgsType>($_self).getInner()
}],
                                          [{
  ([](::mlir::Type inner, mlir::OperandRange::type_range elems) {
    for (auto e : elems) {
      if (!std::equal_to<>()(inner, e)) return false;
    }
    return true;
  })
}]>;

def VarArgsOp : ZmirOp<"vargs", [SameTypeOperands, VarArgsTypeCheckPred]> {
  let summary = "groups variable arguments";

  let arguments = (ins Variadic<AnyNonPendingZmirType>:$elements);
  let results = (outs VarArgs:$result);

  let assemblyFormat = "$elements attr-dict `:` type($result)";
}

def AllocArrayOp : ZmirOp<"alloc_array"> {
  let summary = "creates a new empty array";

  let results = (outs Array:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def ReadArrayOp : ZmirOp<"read_array"> {
  let summary = "read data from an array";

  let arguments = (ins PendingOf<Array>:$lvalue, Variadic<AnyType>:$indices);
  let results = (outs AnyZmirType:$result);

  let assemblyFormat = [{
    $lvalue `[` $indices `]` attr-dict `:` type($lvalue) `[` type($indices) `]` `,` type($result)
  }];
}

def WriteArrayOp : ZmirOp<"write_array"> {
  let summary = "write data into an array";

  let arguments = (ins Array:$array, Variadic<AnyType>:$indices,
      AnyZmirType:$value);
  let assemblyFormat = [{
    $array `[` $indices `]` `=` $value attr-dict `:` type($array) `[` type($indices) `]` `,` type($value)
  }];
}

def GetArrayLenOp : ZmirOp<"array_len"> {
  let summary = "returns the length of an array";

  let arguments = (ins Array:$array);
  let results = (outs Index:$length);
}

def ValToIndexOp : ZmirOp<"toindex"> {
  let summary = "convert val to index";

  let arguments = (ins PendingOf<Val>:$val);
  let results = (outs Index:$result);
}

def IndexToValOp : ZmirOp<"fromindex"> {
  let summary = "convert index to val";

  let arguments = (ins Index:$index);
  let results = (outs Val:$result);
}

def LitValOp : ZmirOp<"literal"> {
  let summary = "a literal value";
  let arguments = (ins UI64Attr:$value);
  let results = (outs Val:$val);

  let assemblyFormat = [{
    $value `:` type($val) attr-dict
  }];
}

def LitStrOp : ZmirOp<"string"> {
  let summary = "a literal value";
  let arguments = (ins StrAttr:$value);
  let results = (outs String:$val);

  let assemblyFormat = [{
    $value `:` type($val) attr-dict
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

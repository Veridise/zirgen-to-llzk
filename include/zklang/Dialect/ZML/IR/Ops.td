// Copyright 2024 Veridise, Inc.

include "mlir/IR/OpBase.td"
include "Dialect.td"
include "Types.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "OpInterfaces.td"

class ZMLOp<string mnemonic, list<Trait> traits = []>
    : Op<ZMLDialect, mnemonic, traits> {}

class ComponentOpBase<string prefix, string bodyFuncName,
                      string constrainFuncName>
    : ZMLOp<prefix#"component", [HasParent<"::mlir::ModuleOp">, Symbol,
                                 SymbolTable, IsolatedFromAbove,
                                 GraphRegionNoTerminator, OpAsmOpInterface,
                                 ComponentInterface]> {
  let summary = prefix#"component declaration";
  let arguments = (ins SymbolNameAttr:$sym_name, UnitAttr:$function,
      UnitAttr:$generic, UnitAttr:$builtin, UnitAttr:$usesBackVariables,
      OptionalAttr<FlatSymbolRefArrayAttr>:$params);
  let regions = (region SizedRegion<1>:$body);

  let builders = [OpBuilder<(ins "::mlir::StringRef":$name,
                      "::mlir::ArrayRef<mlir::NamedAttribute>":$attrs,
                      CArg<"bool", "false">:$usesBackVariables)>,
                  OpBuilder<(ins "::mlir::StringRef":$name,
                      "::zml::IsBuiltIn":$tag)>,
                  OpBuilder<(ins "::mlir::StringRef":$name,
                      "::mlir::ArrayRef<mlir::StringRef>":$params,
                      "::zml::IsBuiltIn":$tag)>,
                  OpBuilder<(ins "::mlir::StringRef":$name,
                      "::mlir::ArrayRef<mlir::StringRef>":$params,
                      "::mlir::ArrayRef<mlir::NamedAttribute>":$attrs,
                      CArg<"bool", "false">:$usesBackVariables)>];

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
    $sym_name (`<` $params^ `>`)?  attr-dict-with-keyword $body
  }];
}

def ComponentOp : ComponentOpBase<"", "body", "body"> {}

def SplitComponentOp : ComponentOpBase<"split_", "compute", "constrain"> {}

def SelfOp : ZMLOp<"self", [NoTerminator]> {
  let summary = "current component scope";
  let description = "Returns the current component after executing its content";

  let results = (outs Component);
  let regions = (region MaxSizedRegion<1>:$body);

  let assemblyFormat = [{
     $body attr-dict `:` type(results) 
  }];

  let skipDefaultBuilders = 1;
  let builders =
      [OpBuilder<(ins "mlir::Type":$compType,
           CArg<"mlir::function_ref<void(mlir::OpBuilder &, mlir::Value)>",
                "nullptr">)>,
       OpBuilder<(ins "mlir::Type":$compType, "mlir::Region &":$region)>];

  let extraClassDeclaration = [{
    mlir::Value getSelfValue() { 
      assert(getRegion().getNumArguments() > 0 && "self region must have one argument");
      return getRegion().getArgument(0); 
    }
  }];
}

//===----------------------------------------------------------------------===//
//  ConstructorRefOp
//===----------------------------------------------------------------------===//

def ConstructorRefOp : ZMLOp<"constructor", [Pure]> {
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

  let builders = [OpBuilder<(ins "ComponentInterface":$comp,
                      "mlir::FunctionType":$fn)>,
                  OpBuilder<(ins "mlir::FlatSymbolRefAttr":$compSym,
                      "mlir::FunctionType":$fn, "bool":$isBuiltin)>];

  let extraClassDeclaration = [{
    /// Returns true if can be built with the given value
    /// and result type.
    static bool isBuildableWith(mlir::Attribute value, mlir::Type type);
  }];

  let hasVerifier = 1;
}

def ExternFnRefOp : ZMLOp<"extern_fn", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs FunctionType);
  let assemblyFormat = "attr-dict $name `:` type(results)";
}

def FieldDefOp : ZMLOp<"field", [HasParent<"::zml::ComponentOp, "
                                           "::zml::SplitComponentOp">,
                                 Symbol]> {
  let summary = "component field definition";
  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<TypeVarOr<Component>>:$type);

  let assemblyFormat = [{
    $sym_name `:` $type attr-dict
  }];
}

def ReadFieldOp
    : ZMLOp<"read_field", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "reads the value of a field";
  let description = [{
    Reads the value of a field in a component.
  }];

  let arguments = (ins TypeVarOr<Component>:$component,
      FlatSymbolRefAttr:$field_name);
  let results = (outs TypeVarOr<Component>:$val);

  let assemblyFormat = [{
    $component `[` $field_name `]` attr-dict `:` type($component) `,` type($val) 
  }];
}

def WriteFieldOp
    : ZMLOp<"write_field", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "writes a value into a field";
  let description = [{
    Writes a value into a field in a component.
  }];

  let arguments = (ins TypeVarOr<Component>:$component,
      FlatSymbolRefAttr:$field_name, TypeVarOr<Component>:$val);

  let assemblyFormat = [{
    $component `[` $field_name `]` `=` $val attr-dict `:` type($component) `,` type($val) 
  }];
}

def SuperCoerceOp : ZMLOp<"super_coerce"> {
  let summary =
      "transitively reads the super field until the target type is reached";

  let arguments = (ins Component:$component);
  let results = (outs TypeVarOr<Component>:$val);

  let assemblyFormat =
      "$component attr-dict `:` type($component) `->` type($val)";
}

def NopOp : ZMLOp<"nop", [Pure]> {

  let arguments = (ins Variadic<AnyZMLType>:$ins);
  let results = (outs Variadic<AnyZMLType>:$outs);

  let assemblyFormat = "($ins `:` type($ins)^ `->`)? type($outs) attr-dict";

  let hasFolder = 1;

  let hasCanonicalizeMethod = 1;
}

def GlobalDefOp : ZMLOp<"global", [HasParent<"::mlir::ModuleOp">]> {
  let summary = "defines a global component";
  let description = "Defines a global component and binds it to a symbol.";

  let arguments = (ins SymbolNameAttr:$sym_name, AnyZMLType:$value,
      TypeAttrOf<AnyZMLType>:$type);

  let assemblyFormat = [{
    $sym_name `=` $value `:` $type `,` type($value) attr-dict
  }];
}

def GetGlobalOp
    : ZMLOp<"get_global", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "loads a global variable";

  let arguments = (ins SymbolNameAttr:$sym_name);

  let assemblyFormat = [{
    $sym_name `:` attr-dict
  }];
}

def ReadBackOp : ZMLOp<"read_back"> {
  let summary = "reads a back variable";

  let arguments = (ins Component:$comp, Component:$distance,
      SymbolNameAttr:$field);
  let results = (outs TypeVarOr<Component>:$result);

  let assemblyFormat =
      "$comp `[` $field `]` `at` $distance `:` type($comp) `,` "
      "type($distance) `,` type($result) attr-dict";
}

def ConstrainCallOp : ZMLOp<"constrain_call"> {
  let summary = "calls the constrain function";

  let arguments = (ins TypeVarOr<Component>:$self, Variadic<AnyType>:$args);
  let assemblyFormat = [{
    $self `(` $args `)` `:`  type($self) (`,` type($args)^ )?  attr-dict
  }];
}

def ConstrainOp : ZMLOp<"constrain"> {
  let summary = "emits a constraint";

  let arguments = (ins TypeVarOr<Component>:$lhs, TypeVarOr<Component>:$rhs);

  let assemblyFormat = [{
  $lhs `=` $rhs attr-dict `:` type($lhs) `,` type($rhs)
  }];
}

def NewArrayOp : ZMLOp<"new_array", [Pure, SameTypeOperands]> {
  let summary = "creates a new array with data";

  let arguments = (ins Variadic<AnyZMLType>:$elements);
  let results = (outs TypeVarOr<Component>:$result);

  let assemblyFormat =
      "$elements attr-dict `:` `[` type($elements) `]` `,` type($result)";
}

def LitValArrayOp : ZMLOp<"litval_array", [Pure, ConstantLike]> {
  let summary = "literal array definition";

  let arguments = (ins DenseI64ArrayAttr:$elements);
  let results = (outs Component:$result);

  let assemblyFormat = "$elements attr-dict `:` type($result)";

  let hasFolder = 1;
}

def VarArgsTypeCheckPred : TypesMatchWith<"inner type matches input operands",
                                          "result", "elements", [{
  ::llvm::cast<::zml::VarArgsType>($_self).getInner()
}],
                                          [{
  ([](::mlir::Type inner, mlir::OperandRange::type_range elems) {
    for (auto e : elems) {
      if (!std::equal_to<>()(inner, e)) return false;
    }
    return true;
  })
}]>;

def VarArgsOp : ZMLOp<"vargs", [SameTypeOperands, VarArgsTypeCheckPred]> {
  let summary = "groups variable arguments";

  let arguments = (ins Variadic<AnyZMLType>:$elements);
  let results = (outs VarArgs:$result);

  let assemblyFormat = "$elements attr-dict `:` type($result)";
}

def AllocArrayOp : ZMLOp<"alloc_array"> {
  let summary = "creates a new empty array";

  let results = (outs Component:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def ReadArrayOp : ZMLOp<"read_array"> {
  let summary = "read data from an array";

  let arguments = (ins TypeVarOr<Component>:$lvalue,
      Variadic<AnyType>:$indices);
  let results = (outs TypeVarOr<Component>:$result);

  let assemblyFormat = [{
    $lvalue `[` $indices `]` attr-dict `:` type($lvalue) `[` type($indices) `]` `,` type($result)
  }];
}

def WriteArrayOp : ZMLOp<"write_array"> {
  let summary = "write data into an array";

  let arguments = (ins TypeVarOr<Component>:$array, Variadic<AnyType>:$indices,
      AnyZMLType:$value, UnitAttr:$computeOnly);
  let assemblyFormat = [{
    $array `[` $indices `]` `=` $value attr-dict `:` type($array) `[` type($indices) `]` `,` type($value)
  }];
}

def GetArrayLenOp : ZMLOp<"array_len", [Pure]> {
  let summary = "returns the length of an array";

  let arguments = (ins TypeVarOr<Component>:$array);
  let results = (outs Index:$length);

  let assemblyFormat = "$array attr-dict `:` type($array)";

  let hasFolder = 1;
}

def ValToIndexOp : ZMLOp<"toindex", [Pure]> {
  let summary = "convert val to index";

  let arguments = (ins TypeVarOr<Component>:$val);
  let results = (outs Index:$result);

  let assemblyFormat = "$val `:` type($val) attr-dict";

  let hasFolder = 1;
}

def ValToI1Op : ZMLOp<"tobool", [Pure]> {
  let summary = "convert val to i1";

  let arguments = (ins TypeVarOr<Component>:$val);
  let results = (outs I1:$result);
  let assemblyFormat = "$val `:` type($val) attr-dict";
}

def IndexToValOp : ZMLOp<"fromindex", [Pure]> {
  let summary = "convert index to val";

  let arguments = (ins Index:$index);
  let results = (outs Component:$result);
  let assemblyFormat = "$index `:` type($result) attr-dict";
}

def LitValOp : ZMLOp<"literal", [Pure, ConstantLike]> {
  let summary = "a literal value";
  let arguments = (ins I64Attr:$value);
  let results = (outs Component:$val);

  let assemblyFormat = [{
    $value `:` type($val) attr-dict
  }];

  let hasFolder = 1;
}

def LitStrOp : ZMLOp<"string", [Pure]> {
  let summary = "a literal value";
  let arguments = (ins StrAttr:$value);
  let results = (outs Component:$val);

  let assemblyFormat = [{
    $value `:` type($val) attr-dict
  }];
}

// Used for the last branch of an if-then-else lowered from a zhl.switch
def AssertOp : ZMLOp<"assert"> {
  let summary = "assert";
  let arguments = (ins AnyType:$cond);

  let assemblyFormat = [{
    $cond  `:` type($cond)  attr-dict 
    }];
}

def LoadValParamOp : ZMLOp<"load_param", [Pure]> {
  let summary = "loads val from parameter";
  let arguments = (ins SymbolNameAttr:$param);
  let results = (outs Component:$val);

  let assemblyFormat = "$param `:` type($val) attr-dict";
}

///////////////////// Builtin Ops /////////////////////

def InRangeOp : ZMLOp<"in_range", [Pure]> {
  let summary = "checks if value is within range";
  let arguments = (ins Component:$low, Component:$mid, Component:$high);
  let results = (outs Component:$out);
  let assemblyFormat =
      [{ $low ` ` `<` `=` $mid ` ` `<` ` ` $high `:` `(` type($low) `,` type($mid) `,` type($high) `)` `->` type($out) attr-dict }];
}

def EqzExtOp : ZMLOp<"eqzext"> {
  let summary = "checks if ext value is zero";
  let arguments = (ins Component:$extval);
  let assemblyFormat = [{ $extval `:` type($extval) attr-dict }];
}

class UnaryOp<string mnemonic, list<Trait> traits = []>
    : ZMLOp<mnemonic, !listconcat([Pure], traits)> {
  let arguments = (ins Component:$in);
  let results = (outs Component:$out);
  let assemblyFormat = [{ $in `:` type($in) `->` type($out) attr-dict }];
}

def IsZeroOp : UnaryOp<"isz"> {}
def NegOp : UnaryOp<"neg"> {}
def InvOp : UnaryOp<"inv"> {}
def ExtInvOp : UnaryOp<"extinv"> {}
def MakeExtOp : UnaryOp<"mkext"> {}

class BinaryOp<string mnemonic, list<Trait> traits = []>
    : ZMLOp<mnemonic, !listconcat([Pure], traits)> {
  let arguments = (ins Component:$lhs, Component:$rhs);
  let results = (outs Component:$out);
  let assemblyFormat =
      [{ $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `->` type($out) attr-dict }];
}

def AddOp : BinaryOp<"add", [Commutative]> {}
def ExtAddOp : BinaryOp<"extadd", [Commutative]> {}
def SubOp : BinaryOp<"sub"> {}
def ExtSubOp : BinaryOp<"extsub"> {}
def MulOp : BinaryOp<"mul", [Commutative]> {}
def ExtMulOp : BinaryOp<"extmul", [Commutative]> {}
def BitAndOp : BinaryOp<"bit_and"> {}
def ModOp : BinaryOp<"mod"> {}

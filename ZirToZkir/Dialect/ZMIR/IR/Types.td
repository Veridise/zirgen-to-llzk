// Copyright 2024 Veridise, Inc.


include "mlir/IR/AttrTypeBase.td"
include "ZirToZkir/Dialect/ZMIR/IR/Dialect.td"

class ZmirType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ZmirDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// XXX: Probably not necessary
// def Expr : ZmirType<"Expr", "expr"> {
//   let summary = "Expression node";
// }

// TODO: Fill these types with what they need.
def Val : ZmirType<"Val", "val"> {
  let summary = "value type";
}

def TypeVar : ZmirType<"TypeVar", "type_var"> {
  let summary = "parametric type linked to a symbol";
}

def String : ZmirType<"String", "string"> {
  let summary = "string type";
}

def Union : ZmirType<"Union", "union"> {
  let summary = "union type";
}

def Component : ZmirType<"Component", "component"> {
  let summary = "component type";
}

def Array : ZmirType<"Array", "array"> {
  let summary = "array type";
}

def AnyZmirType : AnyTypeOf<[Val, TypeVar, String, Union, Component, Array],
                         "A type that can participate in ZMIR components">;

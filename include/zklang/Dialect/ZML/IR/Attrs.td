// Copyright 2024 Veridise, Inc.

include "mlir/IR/AttrTypeBase.td"
include "Dialect.td"

class ZmirAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ZmirDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Variadic : ZmirAttr<"Variadic", "variadic"> {}

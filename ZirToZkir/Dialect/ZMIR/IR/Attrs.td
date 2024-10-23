// Copyright 2024 Veridise, Inc.


include "mlir/IR/AttrTypeBase.td"
include "ZirToZkir/Dialect/ZMIR/IR/Dialect.td"

class ZmirAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ZmirDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

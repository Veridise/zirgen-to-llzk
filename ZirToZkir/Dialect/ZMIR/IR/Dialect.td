// Copyright 2024 Veridise, Inc.

#ifndef ZMIR_DIALECT
#define ZMIR_DIALECT

include "mlir/IR/DialectBase.td"

def ZmirDialect : Dialect {
  let name = "zmir";
  let cppNamespace = "::zkc::Zmir";
  let summary = "Zir to Zkir medium-level dialect";
  let useDefaultTypePrinterParser = 1;

  let dependentDialects = [
    "::mlir::index::IndexDialect",
    "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect"
  ];
}

#endif // ZMIR_DIALECT

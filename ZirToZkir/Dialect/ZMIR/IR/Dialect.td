// Copyright 2024 Veridise, Inc.

#ifndef ZMIR_DIALECT
#define ZMIR_DIALECT

include "mlir/IR/DialectBase.td"

def ZmirDialect : Dialect {
  let name = "zmir";
  let cppNamespace = "::zkc::Zmir";
  let summary = "Zir medium-level dialect";
  let useDefaultTypePrinterParser = 1;
}

#endif // ZMIR_DIALECT

// Copyright 2024 Veridise, Inc.

#ifndef ZML_DIALECT
#define ZML_DIALECT

include "mlir/IR/DialectBase.td"

def ZMLDialect : Dialect {
  let name = "zml";
  let cppNamespace = "::zml";
  let summary = "ZIR medium-level dialect";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;

  let dependentDialects = ["::mlir::index::IndexDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::scf::SCFDialect"];
}

#endif // ZML_DIALECT

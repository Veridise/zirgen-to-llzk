//===- Dialect.td - ZML dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the definition of the ZML dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ZML_DIALECT
#define ZML_DIALECT

include "mlir/IR/DialectBase.td"

def ZMLDialect : Dialect {
  let name = "zml";
  let cppNamespace = "::zml";
  let summary = "Zirgen medium-level dialect";
  let description = [{
    This dialect serves as a stepping stone for lowering from ZHL to LLZK.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;

  let dependentDialects = ["::mlir::index::IndexDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::scf::SCFDialect"];
}

#endif // ZML_DIALECT

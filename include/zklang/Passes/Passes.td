//===- Passes.td - Conversion passes definitions -----------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes definitions for passes that convert between dialects 
// or are part of the broader lowering pipeline of the frontend.
//
//===----------------------------------------------------------------------===//

#ifndef ZMIR_TRANSLATION_PASSES
#define ZMIR_TRANSLATION_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def StripTests : Pass<"strip-tests", "mlir::ModuleOp"> {
  let summary = "Remove the tests from the input program";
  let constructor = "zklang::createStripTestsPass()";

  let dependentDialects = ["zirgen::Zhl::ZhlDialect"];
}

def ConvertZhlToZml : Pass<"lower-zhl", "mlir::ModuleOp"> {
  let summary = "Converts ZHL operations to ZML operations";
  let constructor = "zklang::createConvertZhlToZmlPass()";

  let dependentDialects = ["zirgen::Zhl::ZhlDialect", "zml::ZMLDialect",
                           "mlir::func::FuncDialect", "mlir::scf::SCFDialect",
                           "mlir::index::IndexDialect"];
}

def ConvertZmlToLlzk : Pass<"zml-to-llzk", "mlir::ModuleOp"> {
  let summary = "Converts ZML operations into LLZK operations";
  let constructor = "zklang::createConvertZmlToLlzkPass()";

  let dependentDialects = ["llzk::LLZKDialect", "zml::ZMLDialect",
                           "mlir::func::FuncDialect", "mlir::scf::SCFDialect",
                           "mlir::index::IndexDialect"];

  let options = [Option<"selectedExtValField", "extval-field", "std::string",
                        /*default=*/"\"babybear\"",
                        "The field ExtVal types operate on.">];
}

def InjectLlzkModAttrs : Pass<"inject-llzk-mod-attrs", "mlir::ModuleOp"> {
  let summary = "Adds module the required module attributes for LLZK";
  let constructor = "zklang::createInjectLlzkModAttrsPass()";

  let dependentDialects = ["llzk::LLZKDialect", "zml::ZMLDialect",
                           "mlir::func::FuncDialect", "mlir::scf::SCFDialect",
                           "mlir::index::IndexDialect"];
}

#endif

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

def AnnotateTypecheckZhl : Pass<"annotate-typecheck-zhl", "mlir::ModuleOp"> {
  let summary = "Checks that the input is well typed";
  let constructor = "zklang::createAnnotateTypecheckZhlPass()";

  let dependentDialects = ["zirgen::Zhl::ZhlDialect", "zml::ZMLDialect"];
}

def InstantiatePODBlocks : Pass<"instantiate-pod-blocks", "mlir::ModuleOp"> {
  let summary = "Creates llzk structs for blocks that need concrete data types";
  let constructor = "zklang::createInstantiatePODBlocksPass()";
  let dependentDialects =
      ["llzk::component::StructDialect", "llzk::function::FunctionDialect"];

  let options =
      [Option<"selectedExtValField", "extval-field", "std::string",
              /*default=*/"\"babybear\"",
              "The field ExtVal types operate on (default: babybear)">];
}

def StripTests : Pass<"strip-tests", "mlir::ModuleOp"> {
  let summary = "Remove tests from the input program";
  let constructor = "zklang::createStripTestsPass()";

  let dependentDialects = ["zirgen::Zhl::ZhlDialect"];
}

def ConvertZhlToLlzkStruct
    : Pass<"convert-zhl-to-llzk-struct", "mlir::ModuleOp"> {
  let summary =
      "Converts ZHL component operations into LLZK struct dialect operations.";
  let constructor = "zklang::createConvertZhlToLlzkStructPass()";

  let dependentDialects = [
    "zirgen::Zhl::ZhlDialect", "zml::ZMLDialect",
    "llzk::component::StructDialect"
  ];

  let options =
      [Option<"selectedExtValField", "extval-field", "std::string",
              /*default=*/"\"babybear\"",
              "The field ExtVal types operate on (default: babybear)">];
}

def ConvertZhlToLlzkFelt
    : Pass<"convert-zhl-to-llzk-felt", "llzk::component::StructDefOp"> {
  let summary = "Converts ZHL operations into LLZK felt dialect operations.";
  let constructor = "zklang::createConvertZhlToLlzkFeltPass()";

  let dependentDialects = [
    "zirgen::Zhl::ZhlDialect", "zml::ZMLDialect",
    "llzk::component::StructDialect", "llzk::felt::FeltDialect"
  ];

  let options =
      [Option<"selectedExtValField", "extval-field", "std::string",
              /*default=*/"\"babybear\"",
              "The field ExtVal types operate on (default: babybear)">];
}

def ConvertZhlToZml : Pass<"lower-zhl", "mlir::ModuleOp"> {
  let summary = "Converts ZHL operations to ZML operations";
  let constructor = "zklang::createConvertZhlToZmlPass()";

  let dependentDialects = [
    "zirgen::Zhl::ZhlDialect", "zml::ZMLDialect", "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect", "mlir::index::IndexDialect"
  ];
}

def ConvertZmlToLlzk : Pass<"zml-to-llzk", "mlir::ModuleOp"> {
  let summary = "Converts ZML operations into LLZK operations";
  let constructor = "zklang::createConvertZmlToLlzkPass()";

  let dependentDialects = [
    "llzk::LLZKDialect", "zml::ZMLDialect", "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect", "mlir::index::IndexDialect"
  ];

  let options =
      [Option<"selectedExtValField", "extval-field", "std::string",
              /*default=*/"\"babybear\"",
              "The field ExtVal types operate on (default: babybear)">];
}

def InjectLlzkModAttrs : Pass<"inject-llzk-mod-attrs", "mlir::ModuleOp"> {
  let summary = "Adds to a module the required attributes for LLZK";
  let constructor = "zklang::createInjectLlzkModAttrsPass()";

  let dependentDialects = [
    "llzk::LLZKDialect", "zml::ZMLDialect", "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect", "mlir::index::IndexDialect"
  ];
}

#endif

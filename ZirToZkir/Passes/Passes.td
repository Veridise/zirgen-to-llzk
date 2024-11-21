// Copyright 2024 Veridise, Inc.

#ifndef ZMIR_TRANSLATION_PASSES
#define ZMIR_TRANSLATION_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def StripTests : Pass<"strip-tests", "mlir::ModuleOp"> {
  let summary = "Remove the tests from the input program";
  let constructor = "zkc::createStripTestsPass()";

  let dependentDialects = ["zirgen::Zhl::ZhlDialect"];
}

def StripDirectives : Pass<"strip-directives", "mlir::ModuleOp"> {
  let summary = "Remove compiler directives from the input program";
  let constructor = "zkc::createStripDirectivesPass()";

  let dependentDialects = ["zirgen::Zhl::ZhlDialect"];
}

def TransformComponentDecls
    : Pass<"transform-component-decls", "mlir::ModuleOp"> {
  let summary = "Transforms zhl components into zmir components";
  let constructor = "zkc::createTransformComponentDeclsPass()";

  let dependentDialects = ["zirgen::Zhl::ZhlDialect",
                           "::zkc::Zmir::ZmirDialect",
                           "::mlir::func::FuncDialect",
  ];
}

def ConvertZhlToZmir : Pass<"convert-zhl-to-zmir", "zkc::Zmir::ComponentOp"> {
  let summary =
      "Converts ZHL operations in the body of a component to ZMIR operations";
  let constructor = "zkc::createConvertZhlToZmirPass()";

  let dependentDialects = ["zirgen::Zhl::ZhlDialect", "zkc::Zmir::ZmirDialect",
                           "mlir::func::FuncDialect", "mlir::scf::SCFDialect",
                           "mlir::index::IndexDialect"];
}

#endif

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

def ConvertZhlToZmir : Pass<"lower-zhl", "mlir::ModuleOp"> {
  let summary =
      "Converts ZHL operations in the body of a component to ZMIR operations";
  let constructor = "zkc::createConvertZhlToZmirPass()";

  let dependentDialects = ["zirgen::Zhl::ZhlDialect", "zkc::Zmir::ZmirDialect",
                           "mlir::func::FuncDialect", "mlir::scf::SCFDialect",
                           "mlir::index::IndexDialect"];
}

def ConvertZmirToLlzk : Pass<"zmir-to-llzk", "llzk::StructDefOp"> {
  let summary =
      "Converts ZMIR operations in the body of a component to LLZK operations";
  let constructor = "zkc::createConvertZmirToLlzkPass()";

  let dependentDialects = ["llzk::LLZKDialect", "zkc::Zmir::ZmirDialect",
                           "mlir::func::FuncDialect", "mlir::scf::SCFDialect",
                           "mlir::index::IndexDialect"];
}

def ConvertZmirComponentsToLlzk
    : Pass<"zmir-components-to-llzk", "mlir::ModuleOp"> {
  let summary = "Converts ZMIR component operations to LLZK operations";
  let constructor = "zkc::createConvertZmirComponentsToLlzkPass()";

  let dependentDialects = ["llzk::LLZKDialect", "zkc::Zmir::ZmirDialect",
                           "mlir::func::FuncDialect", "mlir::scf::SCFDialect",
                           "mlir::index::IndexDialect"];
}

#endif

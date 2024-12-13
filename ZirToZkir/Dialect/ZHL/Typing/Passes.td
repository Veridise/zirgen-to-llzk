// Copyright 2024 Veridise, Inc.

#ifndef ZMIR_TRANSFORM_PASSES
#define ZMIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def PrintTypeBindings : Pass<"zhl-print-type-bindings", "mlir::ModuleOp"> {
  let summary = "Print the type bindings inferred for the components";
  let constructor = "zhl::createPrintTypeBindingsPass()";

  let dependentDialects = ["::zirgen::Zhl::ZhlDialect"];
}

#endif

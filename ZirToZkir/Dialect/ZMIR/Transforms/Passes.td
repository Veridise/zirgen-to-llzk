// Copyright 2024 Veridise, Inc.

#ifndef ZMIR_TRANSFORM_PASSES
#define ZMIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def InjectBuiltIns : Pass<"inject-builtins", "mlir::ModuleOp"> {
  let summary = "Add to the root module the ZIR language builtins";
  let constructor = "zkc::Zmir::createInjectBuiltInsPass()";

  let dependentDialects = [
    "::zkc::Zmir::ZmirDialect",
    "::mlir::func::FuncDialect",
  ];
}

#endif

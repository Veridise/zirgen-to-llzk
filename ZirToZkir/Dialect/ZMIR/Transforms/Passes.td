// Copyright 2024 Veridise, Inc.

#ifndef ZMIR_TRANSFORM_PASSES
#define ZMIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def InjectBuiltIns : Pass<"inject-builtins", "mlir::ModuleOp"> {
  let summary = "Add to the root module the ZIR language builtins";
  let constructor = "zkc::Zmir::createInjectBuiltInsPass()";

  let dependentDialects = ["::zkc::Zmir::ZmirDialect",
                           "::mlir::func::FuncDialect",
  ];
}

/*
def SplitComponentBody
    : Pass<"split-component-body", "::zkc::Zmir::ComponentOp"> {
  let summary = "Splits the body function of a component into compute and "
                "constrain functions in preparation for lowering to ZKIR";
  let constructor = "zkc::Zmir::createSplitComponentBodyPass()";

  let dependentDialects = ["::zkc::Zmir::ZmirDialect",
                           "::mlir::func::FuncDialect"];
}
*/
#endif

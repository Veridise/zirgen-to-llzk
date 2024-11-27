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

def LowerBuiltIns : Pass<"lower-builtins", "mlir::func::FuncOp"> {
  let summary = "transforms calls to builtin components to 'native' ops";
  let constructor = "zkc::Zmir::createLowerBuiltInsPass()";

  let dependentDialects = ["::zkc::Zmir::ZmirDialect",
                           "::mlir::func::FuncDialect"];
}

def RemoveBuiltIns : Pass<"remove-builtins", "mlir::ModuleOp"> {
  let summary = "removes the builtin component definitions that can be lowered";
  let constructor = "zkc::Zmir::createRemoveBuiltInsPass()";

  let dependentDialects = ["::zkc::Zmir::ZmirDialect",
                           "::mlir::func::FuncDialect"];
}

def SplitComponentBody : Pass<"split-component-body", "mlir::ModuleOp"> {
  let summary = "Splits the body function of a component into compute and "
                "constrain functions in preparation for lowering to ZKIR";
  let constructor = "zkc::Zmir::createSplitComponentBodyPass()";

  let dependentDialects = ["::zkc::Zmir::ZmirDialect",
                           "::mlir::func::FuncDialect"];
}

class RemoveIllegalOps<string func>
    : Pass<"remove-illegal-"#!tolower(func)#"-ops", "mlir::func::FuncOp"> {

  let summary =
      "Removes from the @"#!tolower(func)#" function of a component any ops "
                                          "that are not allowed";
  let constructor = "zkc::Zmir::createRemoveIllegal"#func#"OpsPass()";

  let dependentDialects = ["zkc::Zmir::ZmirDialect",
                           "::mlir::func::FuncDialect"];
}

def RemoveIllegalComputeOps : RemoveIllegalOps<"Compute"> {}

def RemoveIllegalConstrainOps : RemoveIllegalOps<"Constrain"> {}

def InsertTemporaries : Pass<"insert-temporaries", "zkc::Zmir::ComponentOp"> {
  let summary = "creates fields for temporary values generated from "
                "constructor calls that do not write into a field";

  let constructor = "zkc::Zmir::createInsertTemporariesPass()";

  let dependentDialects = ["zkc::Zmir::ZmirDialect", "mlir::func::FuncDialect"];
}

def LegalizeTypes : Pass<"legalize-types", "zkc::Zmir::ComponentOp"> {
  let summary = "removes pending types";

  let constructor = "zkc::Zmir::createLegalizeTypesPass()";

  let dependentDialects = ["zkc::Zmir::ZmirDialect", "mlir::func::FuncDialect"];
}

def ExpandSuperCoercion : Pass<"super-coercion", "zkc::Zmir::ComponentOp"> {
  let summary = "transitively coerces components types into their super type";

  let constructor = "zkc::Zmir::createExpandSuperCoercionPass()";

  let dependentDialects = ["zkc::Zmir::ZmirDialect", "mlir::func::FuncDialect"];
}

#endif

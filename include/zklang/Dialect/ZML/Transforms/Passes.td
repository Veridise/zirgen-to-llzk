//===- Passes.td - ZML transformation passes definitions ---*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes passes for transforming circuits defined with the
// ZML dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ZML_TRANSFORM_PASSES
#define ZML_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def InjectBuiltIns : Pass<"inject-builtins", "mlir::ModuleOp"> {
  let summary = "Add to the root module the zirgen language builtins";
  let constructor = "zml::createInjectBuiltInsPass()";

  let dependentDialects = [
    "::zml::ZMLDialect",
    "::mlir::func::FuncDialect",
  ];
}

def LowerBuiltIns : Pass<"lower-builtins", "mlir::func::FuncOp"> {
  let summary = "transforms calls to builtin components to operations";
  let constructor = "zml::createLowerBuiltInsPass()";

  let dependentDialects = ["::zml::ZMLDialect", "::mlir::func::FuncDialect"];
}

def RemoveBuiltIns : Pass<"remove-builtins", "mlir::ModuleOp"> {
  let summary = "removes the builtin component definitions that can be lowered";
  let constructor = "zml::createRemoveBuiltInsPass()";

  let dependentDialects = ["::zml::ZMLDialect", "::mlir::func::FuncDialect"];
}

def SplitComponentBody : Pass<"split-component-body", "mlir::ModuleOp"> {
  let summary = "Splits the body function of a component into compute and "
                "constrain functions in preparation for lowering to LLZK";
  let constructor = "zml::createSplitComponentBodyPass()";

  let dependentDialects = ["::zml::ZMLDialect", "::mlir::func::FuncDialect"];
}

class RemoveIllegalOps<string func>
    : Pass<"remove-illegal-" #!tolower(func) #"-ops", "mlir::func::FuncOp"> {

  // "@" must be escaped for proper markdown creation
  let summary = "Removes any ops that not allowed in the `\\@" #!tolower(
      func) #"` function";
  let constructor = "zml::createRemoveIllegal" #func #"OpsPass()";

  let dependentDialects = ["zml::ZMLDialect", "::mlir::func::FuncDialect"];
}

def RemoveIllegalComputeOps : RemoveIllegalOps<"Compute"> {}

def RemoveIllegalConstrainOps : RemoveIllegalOps<"Constrain"> {}

#endif

//===- Passes.td - ZHL utility passes definitions ----------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a pass that prints all the deduced types during type
// analysis.
//
//===----------------------------------------------------------------------===//

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

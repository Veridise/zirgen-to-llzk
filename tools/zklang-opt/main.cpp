//===- main.cpp - zklang-opt ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Dialect/InitDialects.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>
#include <tools/config.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZHL/Typing/Passes.h>
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/Transforms/Passes.h>
#include <zklang/Passes/Passes.h>

int main(int argc, char **argv) {
  zklang::configureTool();
  mlir::DialectRegistry registry;

  mlir::registerCSEPass();
  mlir::registerCanonicalizer();

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createReconcileUnrealizedCastsPass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createTopologicalSortPass();
  });
  zklang::registerPasses();
  zml::registerPasses();
  zhl::registerPasses();

  registry.insert<zml::ZMLDialect>();
  registry.insert<zirgen::Zhl::ZhlDialect>();
  llzk::registerAllDialects(registry);
  return failed(mlir::MlirOptMain(argc, argv, "zklang optimizer\n", registry));
}

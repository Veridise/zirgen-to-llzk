//===- main.cpp - zklang frontend -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <mlir/Support/LogicalResult.h>
#include <tools/config.h>
#include <zklang/Frontends/ZIR/Driver.h>

int main(int argc, char **argv) {
  zklang::configureTool();
  return mlir::failed(zklang::zirDriver(argc, argv));
}

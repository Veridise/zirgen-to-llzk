//===- Patterns.h - ExtVal lowering patterns --------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a function to populate a rewrite pattern set with
// patterns to lower ExtVal operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zklang/Dialect/ZML/ExtVal/Conversion.h>
#include <zklang/Dialect/ZML/IR/Ops.h>

namespace zml {

void populateExtValToLlzkConversionPatterns(mlir::RewritePatternSet &patterns, const mlir::TypeConverter &tc, mlir::MLIRContext *ctx, const extval::BaseConverter &);

} // namespace zml

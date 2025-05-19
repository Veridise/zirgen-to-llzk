//===- Patterns.h - ZHL->LLZK felt conversion patterns ----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file exposes functions for loading dialect conversion from
// ZHL to LLZK felt.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class TypeConverter;
class ConversionTarget;
class RewritePatternSet;
} // namespace mlir

namespace zklang {

void populateZhlToLlzkFeltConversionPatterns(const mlir::TypeConverter &, mlir::RewritePatternSet &);
void populateZhlToLlzkFeltConversionTarget(mlir::ConversionTarget &);
void populateZhlToLlzkFeltConversionPatternsAndLegality(const mlir::TypeConverter &, mlir::RewritePatternSet &, mlir::ConversionTarget &);

} // namespace zklang

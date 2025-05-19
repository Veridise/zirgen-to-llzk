//===- Patterns.h - ZHL->LLZK struct conversion patterns --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file exposes functions for loading dialect conversion from
// ZHL to LLZK struct.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class TypeConverter;
class ConversionTarget;
class RewritePatternSet;
} // namespace mlir

namespace zklang {

/// Populates the pattern set with all the patterns that can convert a zhl operation into a llzk's
/// struct dialect operation.
void populateZhlToLlzkStructConversionPatterns(const mlir::TypeConverter &, mlir::RewritePatternSet &);

/// Populates the conversion target with all the dialects and ops related to converting a zhl
/// operation into a llzk's struct dialect operation.
void populateZhlToLlzkStructConversionTarget(mlir::ConversionTarget &);

/// Populates the pattern set with all the patterns that can convert a zhl operation into a llzk's
/// struct dialect operation and the conversion targetwith all the dialects and ops related to
/// converting a zhl operation into a llzk's struct dialect operation.
void populateZhlToLlzkStructConversionPatternsAndLegality(const mlir::TypeConverter &, mlir::RewritePatternSet &, mlir::ConversionTarget &);

/// Populates the pattern set with all the patterns that can convert a zhl ComponentOp operation
/// into a llzk's struct dialect operation.
void populateZhlComponentToLlzkStructConversionPatterns(const mlir::TypeConverter &, mlir::RewritePatternSet &);

/// Populates the conversion target with all the dialects and ops related to converting a zhl
/// ComponentOp operation into a llzk's struct dialect operation.
void populateZhlComponentToLlzkStructConversionTarget(mlir::ConversionTarget &);

/// Populates the pattern set with all the patterns that can convert a zhl ComponentOp operation
/// into a llzk's struct dialect operation and the conversion target with all the dialects and ops
/// related to converting a zhl ComponentOp operation into a llzk's struct dialect operation.
void populateZhlComponentToLlzkStructConversionPatternsAndLegality(const mlir::TypeConverter &, mlir::RewritePatternSet &, mlir::ConversionTarget &);

} // namespace zklang

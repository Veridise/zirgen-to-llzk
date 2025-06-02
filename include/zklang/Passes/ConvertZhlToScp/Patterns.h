//===- Patterns.h - ZHL->SCP conversion patterns ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file exposes functions for loading dialect conversion from
// ZHL to the standard SCP dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class TypeConverter;
class ConversionTarget;
class RewritePatternSet;
} // namespace mlir

namespace zklang {

/// Populates the pattern set with all the patterns that can convert a zhl operation into a
/// scp dialect operation.
void populateZhlToScpConversionPatterns(const mlir::TypeConverter &, mlir::RewritePatternSet &);

/// Populates the conversion target with all the dialects and ops related to converting a zhl
/// operation into a scp dialect operation.
void populateZhlToScpConversionTarget(mlir::ConversionTarget &);

/// Populates the pattern set with all the patterns that can convert a zhl operation into a
/// scp dialect operation and the conversion targetwith all the dialects and ops related to
/// converting a zhl operation into a scp dialect operation.
inline void populateZhlToScpConversionPatternsAndLegality(
    const mlir::TypeConverter &tc, mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target
) {
  populateZhlToScpConversionPatterns(tc, patterns);
  populateZhlToScpConversionTarget(target);
}

} // namespace zklang

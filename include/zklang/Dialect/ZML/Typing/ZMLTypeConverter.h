//===- ZMLTypeConverter.h - ZML type conversion -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a type converter sub class that handles type conversion
// during the ZHl to ZML conversion.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Transforms/DialectConversion.h>

/// Climbs the use def chain until it finds a
/// value that is not defined by an unrealized cast
/// and that is a legal type. Returns that type.
/// If that doesn't happen returns the initial type.
mlir::Value findTypeInUseDefChain(mlir::Value v, const mlir::TypeConverter *converter);
void findTypesInUseDefChain(
    mlir::ValueRange r, const mlir::TypeConverter *converter,
    llvm::SmallVector<mlir::Value> &results
);

namespace zml {

class ZMLTypeConverter : public mlir::TypeConverter {
public:
  ZMLTypeConverter();
};

} // namespace zml

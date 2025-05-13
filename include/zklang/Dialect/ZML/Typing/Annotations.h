//===- Annotations.h - Type binding attr annotation -------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes functions for annotating MLIR ops with type bindings.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace zhl {
class ZIRTypeAnalysis;
}

namespace zml {

/// Adds FixedTypeBindingAttr attributes to the operations for which
/// the given analysis was performed.
void annotateOperations(mlir::MLIRContext *, mlir::ModuleOp, const zhl::ZIRTypeAnalysis &);

} // namespace zml

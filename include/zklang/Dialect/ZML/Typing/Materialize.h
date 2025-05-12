//===- Materialize.h - Type binding to MLIR type ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes functions for converting type bindings into MLIR types.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

namespace zml {

mlir::Type materializeTypeBinding(mlir::MLIRContext *, const zhl::TypeBinding &);
mlir::FunctionType
materializeTypeBindingConstructor(mlir::Builder &, const zhl::TypeBinding &, const zhl::TypeBindings &);

} // namespace zml

//===- Interpreter.h - Compile-time interpreter -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a set of functions that interpret compile-time constant
// expressions in the ZHL dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Support/LogicalResult.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

namespace zhl::expr::interpreter {

/// Interpret an operation and return a new binding with a constant expression if one could be
/// computed or with the expression removed if either the interpreter failed or the op does not
/// support constant expressions.
/// By default removes any expression that may be in the binding. To handle ops that may return a
/// constant expression create specializations of this template.
template <typename Op, typename... Args>
TypeBinding interpretOp(Op, const TypeBinding &binding, Args &&...) {
  return TypeBinding::NoExpr(binding);
}

template <> TypeBinding interpretOp(zirgen::Zhl::TypeParamOp, const TypeBinding &);

template <> TypeBinding interpretOp(zirgen::Zhl::LiteralOp, const TypeBinding &);

template <>
TypeBinding
interpretOp(zirgen::Zhl::ConstructOp, const TypeBinding &, mlir::ArrayRef<TypeBinding> &&);

template <typename Op, typename... Args>
mlir::FailureOr<TypeBinding>
interpretOp(Op op, const mlir::FailureOr<TypeBinding> &binding, Args &&...args) {
  if (mlir::failed(binding)) {
    return mlir::failure();
  }
  return interpretOp(op, *binding, std::forward<Args>(args)...);
}

} // namespace zhl::expr::interpreter

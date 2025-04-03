//===- Specialization.h - Generic types specialization  ---------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes functions for specializing generic types into concrete
// types by replacing generic parameters for concrete types.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/StringSet.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

namespace zhl {

/// Keeps track of the parameter mappings at increasingly deep nesting levels
/// while specializing a TypeBinding.
class ParamsScopeStack {
public:
  explicit ParamsScopeStack(const Params &root);
  const TypeBinding *operator[](mlir::StringRef name);
  void pushScope(const Params &param);
  void popScope();
  void print(llvm::raw_ostream &os) const;

private:
  std::vector<const Params *> stack;
};

/// Given a stack of type parameter mappings specialize (alpha-reduce) the TypeBinding replacing
/// type variables with the associated types in the mapping. Replaces types in the generic params
/// mapping, the constructor argument types, types of members and the super type. If there exist
/// type variables in the scope that should be left as is they need to be added to the set of free
/// variables. This function edits the TypeBinding in place and returns mlir::failure() in case of
/// error.
mlir::LogicalResult
specializeTypeBinding(TypeBinding *dst, ParamsScopeStack &scopes, const llvm::StringSet<> &FV, const TypeBindings &);
mlir::LogicalResult
specializeTypeBinding(TypeBinding *dst, ParamsScopeStack &scopes, const TypeBindings &);

} // namespace zhl

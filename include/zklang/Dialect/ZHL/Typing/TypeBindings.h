//===- TypeBindings.h - Type bindings factory -------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the TypeBindings that implements factory methods for
// creating type bindings.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <cstdint>
#include <deque>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/Expr.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>

namespace llvm {
class raw_ostream;
}

namespace zhl {

class TypeBindings {
public:
  explicit TypeBindings(mlir::Location);

  const TypeBinding &Component();
  const TypeBinding &Component() const;
  const TypeBinding &Bottom() const;
  TypeBinding Const(uint64_t value, mlir::Location loc) const;
  TypeBinding UnkConst(mlir::Location loc) const;
  TypeBinding Array(TypeBinding type, uint64_t size, mlir::Location loc) const;
  TypeBinding Array(TypeBinding type, TypeBinding size, mlir::Location loc) const;
  TypeBinding UnkArray(TypeBinding type, mlir::Location loc) const;

  TypeBinding Const(uint64_t value) const;
  TypeBinding UnkConst() const;
  TypeBinding Array(TypeBinding type, uint64_t size) const;
  TypeBinding Array(TypeBinding type, TypeBinding size) const;
  TypeBinding UnkArray(TypeBinding type) const;

  [[nodiscard]] bool Exists(mlir::StringRef name) const;

  template <typename... Args>
  const TypeBinding &Create(mlir::StringRef name, mlir::Location loc, Args &&...args) {
    return insert(name, TypeBinding(name, loc, std::forward<Args>(args)...));
  }

  template <typename... Args> const TypeBinding &Create(mlir::StringRef name, Args &&...args) {
    return Create(name, unk, std::forward<Args>(args)...);
  }

  /// Creates a type binding and keeps track of its memory, but it is not registered in the
  /// named bindings table.
  template <typename... Args>
  const TypeBinding &CreateAnon(mlir::StringRef name, mlir::Location loc, Args &&...args) {
    return Manage(TypeBinding(nameUniquer.getName(name), loc, std::forward<Args>(args)...));
  }

  template <typename... Args> const TypeBinding &CreateAnon(mlir::StringRef name, Args &&...args) {
    return CreateAnon(name, unk, std::forward<Args>(args)...);
  }

  template <typename... Args>
  const TypeBinding &CreateBuiltin(mlir::StringRef name, mlir::Location loc, Args &&...args) {
    return insert(name, TypeBinding(name, loc, std::forward<Args>(args)..., Frame(), true));
  }

  template <typename... Args>
  const TypeBinding &CreateBuiltin(mlir::StringRef name, Args &&...args) {
    return CreateBuiltin(name, unk, std::forward<Args>(args)...);
  }

  [[nodiscard]] const TypeBinding &Get(mlir::StringRef name) const;
  [[nodiscard]] mlir::FailureOr<TypeBinding> MaybeGet(mlir::StringRef name) const;
  [[nodiscard]] TypeBinding &Manage(const TypeBinding &) const;

private:
  const TypeBinding &insert(mlir::StringRef, TypeBinding &&);

  template <typename... Args>
  TypeBinding makeBuiltin(mlir::StringRef name, mlir::Location loc, Args &&...args) {
    return TypeBinding(name, loc, std::forward<Args>(args)..., Frame(), true);
  }

  template <typename... Args> TypeBinding makeBuiltin(mlir::StringRef name, Args &&...args) {
    return makeBuiltin(name, unk, std::forward<Args>(args)...);
  }

  /// Keeps track of how many anonymous type names have been created
  /// and returns an unique name every time.
  class NameUniquer {
    llvm::StringSet<> seen;

  public:
    std::string getName(llvm::StringRef originalName);
  };

  NameUniquer nameUniquer;
  mlir::Location unk;
  llvm::StringMap<TypeBinding> bindings;
  /// This deque is used as an allocator to hold pointers to type bindings that other type bindings
  /// can safely use as reference. Is used by the Manage member function but since that function is
  /// const this member variable has to be mutable to be able to write in it.
  mutable std::deque<TypeBinding> managedBindings;
  TypeBinding bottom;
};

} // namespace zhl

//===- TypeBindingRef.h - Type binding reference ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the TypeBindingRef class.
//
//===----------------------------------------------------------------------===//

#pragma once

// #include <cassert>
// #include <cstdint>
// #include <llvm/ADT/Bitset.h>
// #include <llvm/ADT/StringMap.h>
// #include <llvm/ADT/StringRef.h>
// #include <llvm/Support/Debug.h>
// #include <memory>
// #include <mlir/IR/Builders.h>
// #include <mlir/IR/Diagnostics.h>
// #include <mlir/IR/DialectImplementation.h>
// #include <mlir/IR/Location.h>
// #include <mlir/Support/LLVM.h>
// #include <mlir/Support/LogicalResult.h>
// #include <zklang/Dialect/ZHL/Typing/Expr.h>
// #include <zklang/Dialect/ZHL/Typing/Frame.h>
// #include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
// #include <zklang/Dialect/ZHL/Typing/Params.h>
// #include <zklang/Support/CopyablePointer.h>

namespace llvm {
class raw_ostream;
class hash_code;
} // namespace llvm

namespace mlir {
class Diagnostic;
}

namespace zhl {

class TypeBinding;

/// Trivially copyable wrapper of a const TypeBinding.
class TypeBindingRef {
public:
  TypeBindingRef(const TypeBinding &Binding) : binding(&Binding) {}

  const TypeBinding &ref() const { return *binding; }
  const TypeBinding *ptr() const { return binding; }

  const TypeBinding &operator*() const { return ref(); }
  const TypeBinding *operator->() const { return ptr(); }

  /// Forwards the equality to the wrapped bindings.
  bool operator==(const TypeBindingRef &) const;

private:
  const TypeBinding *binding;
};

mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, const TypeBindingRef &b);
llvm::hash_code hash_value(const TypeBindingRef &);

} // namespace zhl

namespace llvm {

raw_ostream &operator<<(raw_ostream &os, const zhl::TypeBindingRef &b);

} // namespace llvm

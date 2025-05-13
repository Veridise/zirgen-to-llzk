//===- COW.h - Copy-on-write storage ----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a structure that implements copy-on-write
// semantics by copying and updating the underlying pointer when a mutable
// reference is accessed.
//
// Unfortunately, nothing in this file says "moo".
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <utility>

namespace zklang {

/// A factory class that always returns null
template <typename Typ> class NullptrFactory {
public:
  static inline std::shared_ptr<Typ> init() { return nullptr; }
};

/// A copy-on-write structure that duplicates the underlying pointer when
/// a mutable reference/pointer is requested.
template <typename Typ, typename Factory = NullptrFactory<Typ>> class COW {
public:
  using self = COW<Typ, Factory>;
  using type = Typ;
  using raw_pointer = type *;
  using pointer = std::shared_ptr<type>;
  using ref = Typ &;
  using cref = const Typ &;

  /// Default initialize the pointer to the result of the Factory.
  COW() : ptr(Factory::init()) {}

  /// Initializes the pointer with the given pointer and adopts ownership of it.
  explicit COW(type &&val) : ptr(std::make_shared<type>(std::move(val))) {}
  explicit COW(const type &val) : ptr(std::make_shared<type>(val)) {}
  explicit COW(raw_pointer Ptr) : ptr(Ptr) {}
  explicit COW(pointer Ptr) : ptr(Ptr) {}
  COW &operator=(pointer other) {
    safeAssign(other);
    return *this;
  }

  /// @brief Initializes the pointer with a shared reference from other.
  COW(const self &other) : ptr(other.ptr) {}
  COW &operator=(const self &other) {
    if (ptr != other.ptr) {
      safeAssign(other.ptr);
    }
    return *this;
  }

  COW(self &&other) : ptr(other.ptr) { other.ptr = Factory::init(); }
  COW &operator=(self &&other) {
    if (this != &other && safeAssign(other.ptr)) {
      other.ptr = Factory::init();
    }
    return *this;
  }

  inline pointer get() {
    detach();
    return ptr;
  }

  inline const pointer get() const { return ptr; }

  inline void set(raw_pointer Ptr) { ptr.reset(Ptr); }
  inline void set(pointer Ptr) { safeAssign(Ptr); }
  inline void set(const type &val) { safeAssign(std::make_shared<type>(val)); }
  inline void set(type &&val) { safeAssign(std::make_shared<type>(std::move(val))); }

  inline pointer operator->() {
    detach();
    return ptr;
  }

  inline const pointer operator->() const { return ptr; }

  inline ref operator*() {
    assert(ptr != nullptr);
    detach();
    return *ptr;
  }

  inline cref operator*() const {
    assert(ptr != nullptr);
    return *ptr;
  }

  inline operator bool() const { return ptr != nullptr; }

  /// Referential equality
  inline bool operator==(const self &other) const { return ptr == other.ptr; }

private:
  static inline pointer safeCopy(const pointer other) {
    return other ? std::make_shared<type>(*other) : Factory::init();
  }

  /// @brief Copy the data in the current shared pointer and create a new pointer
  /// for it. Used when a mutable reference/pointer is requested and a write may occur.
  inline void detach() {
    if (ptr) {
      ptr = std::make_shared<type>(*ptr);
    }
  }

  /// @brief Assign only if the current pointer is not equal to the `other` pointer.
  inline bool safeAssign(pointer other) {
    if (ptr == other) {
      return false;
    }
    ptr = other;
    return true;
  }

  pointer ptr;
};

} // namespace zklang

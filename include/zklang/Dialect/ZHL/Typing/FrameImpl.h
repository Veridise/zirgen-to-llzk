//===- FrameImpl.h - allocateSlot implementation ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the implementation of FrameInfo::allocateSlot to break
// a cyclic dependency.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/Support/Debug.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>

namespace zhl {
namespace detail {

template <typename Slot, typename... Args> FrameSlot *FrameInfo::allocateSlot(Args &&...args) {
  FrameSlot *slot = new Slot(std::forward<Args>(args)...);
  nCreatedSlots++;
  slots.push_back(*slot);
  slot->setParent(this);
  return slot;
}

} // namespace detail
} // namespace zhl

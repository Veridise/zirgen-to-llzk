//===- FrameInfo.h - Inner frame implementation -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the internal implementation of a frame.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/simple_ilist.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>

namespace llvm {
class raw_ostream;
}

namespace zhl {

class Frame;

namespace detail {

/// Inner implementation of a Frame. It is shared via a pointer by a group a Frame instances.
class FrameInfo {
  using SlotsList = llvm::simple_ilist<FrameSlot>;

public:
  ~FrameInfo();
  template <typename Slot, typename... Args> FrameSlot *allocateSlot(Args &&...args) {
    FrameSlot *slot = new Slot(std::forward<Args>(args)...);
    nCreatedSlots++;
    slots.push_back(*slot);
    slot->setParent(this);
    return slot;
  }

  using iterator = SlotsList::iterator;
  using const_iterator = SlotsList::const_iterator;

  iterator begin();
  const_iterator begin() const;
  iterator end();
  const_iterator end() const;

  void setParentSlot(FrameSlot *);
  FrameSlot *getParentSlot() const;

  void print(llvm::raw_ostream &) const;

private:
  FrameSlot *parent = nullptr;
  SlotsList slots;
  size_t nCreatedSlots = 0;
};

} // namespace detail

} // namespace zhl

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
#include <llvm/Support/raw_ostream.h>

namespace zhl {

class Frame;
class FrameSlot;

namespace detail {

/// Inner implementation of a Frame. It is shared via a pointer by a group a Frame instances.
class FrameInfo {
public:
  ~FrameInfo();
  template <typename Slot, typename... Args> FrameSlot *allocateSlot(Args &&...args);

  using SlotsList = llvm::simple_ilist<FrameSlot>;

  SlotsList::iterator begin();
  SlotsList::const_iterator begin() const;
  SlotsList::iterator end();
  SlotsList::const_iterator end() const;

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

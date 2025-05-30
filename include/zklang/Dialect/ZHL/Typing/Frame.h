//===- Frame.h - Memory frames for components -------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a class that represents the memory used by a component
// during runtime.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/simple_ilist.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <zklang/Dialect/ZHL/Typing/FrameInfo.h>

namespace zhl {

class FrameSlot;

// Represents the memory used by a component. Collects slots that contain
// components or other frames. This information is later used to generate
// field reads and writes in the LLZK structs for the operations that are
// needed in the constrain function but cannot be generated there.
class Frame {
public:
  Frame();
  Frame(const Frame &);
  Frame(Frame &&);
  Frame &operator=(const Frame &);
  Frame &operator=(Frame &&);

  template <typename Slot, typename... Args> Slot *allocateSlot(Args &&...args) {
    return static_cast<Slot *>(info->allocateSlot<Slot, Args...>(std::forward<Args>(args)...));
  }

  using iterator = detail::FrameInfo::iterator;
  using const_iterator = detail::FrameInfo::const_iterator;

  iterator begin();
  const_iterator begin() const;
  iterator end();
  const_iterator end() const;

  // The frame information forms a tree of slots.
  // Thus, each frame may be contained inside a slot of a frame in an upper level of the hierarchy.
  // Once set, overwriting the parent is considered a bug.
  void setParentSlot(FrameSlot *);
  FrameSlot *getParentSlot() const;

  friend FrameSlot;

  void print(llvm::raw_ostream &) const;

private:
  // A pointer to the unique information about the frame
  std::shared_ptr<detail::FrameInfo> info;
};

} // namespace zhl

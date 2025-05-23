//===- FrameSlot.h - Base class for slots -----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the base class for frame slots that allocate memory
// inside a component's frame.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/ilist_node.h>
#include <llvm/ADT/ilist_node_options.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Support/LLVM.h>

namespace zhl {

namespace detail {
class FrameInfo;
}

class Frame;
class FrameRef;

/// Base class for anything that can be allocated in a frame.
class FrameSlot : public llvm::ilist_node_with_parent<FrameSlot, detail::FrameInfo> {
public:
  enum FrameSlotKind { FS_Component, FS_Array, FS_Frame, FS_ComponentEnd };

  virtual ~FrameSlot() = default;
  /// Returns the name of the slot, or "$temp" if the slot is nameless.
  /// When creating fields in the component the symbol table will take care of renaming them to a
  /// non-conflicting name.
  virtual mlir::StringRef getSlotName() const;

  /// Returns true if the slot was not given an explicit name.
  virtual bool isTemporary() const;

  virtual void rename(llvm::StringRef);

  FrameSlotKind getKind() const;

  void setParent(const detail::FrameInfo *);
  FrameSlot *getParentSlot() const;
  bool belongsTo(const Frame &) const;

  friend llvm::ilist_node_with_parent<FrameSlot, detail::FrameInfo>;

  virtual void print(llvm::raw_ostream &) const;

protected:
  FrameSlot(FrameSlotKind);
  FrameSlot(FrameSlotKind, mlir::StringRef);

  virtual mlir::StringRef defaultNameForTemporaries() const;

private:
  const detail::FrameInfo *getParent() const;

  const FrameSlotKind kind;
  llvm::SmallString<10> name;
  const detail::FrameInfo *parentFrame;
};

} // namespace zhl

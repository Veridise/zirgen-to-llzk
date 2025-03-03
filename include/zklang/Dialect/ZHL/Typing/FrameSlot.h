#pragma once

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/ilist_node.h>
#include <llvm/ADT/ilist_node_options.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/FrameInfo.h>

namespace zhl {

namespace detail {
class FrameInfo;
}

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

  virtual void rename(llvm::StringRef);

  FrameSlotKind getKind() const;

  void setParent(const detail::FrameInfo *);
  FrameSlot *getParentSlot() const;
  bool belongsTo(const Frame &) const;

  friend llvm::ilist_node_with_parent<FrameSlot, detail::FrameInfo>;

protected:
  FrameSlot(FrameSlotKind);
  FrameSlot(FrameSlotKind, mlir::StringRef);

private:
  const detail::FrameInfo *getParent() const;

  const FrameSlotKind kind;
  llvm::SmallString<10> name;
  const detail::FrameInfo *parentFrame;
};

} // namespace zhl

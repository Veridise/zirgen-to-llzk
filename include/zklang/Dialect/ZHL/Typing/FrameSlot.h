#pragma once

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/ilist_node.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>

namespace zhl {

namespace detail {
class FrameInfo;
}
class TypeBinding;

/// Base class for anything that can be allocated in a frame.
class FrameSlot : public llvm::ilist_node<FrameSlot> {
public:
  enum FrameSlotKind { FS_Array, FS_Component, FS_Frame };

  virtual ~FrameSlot() = default;
  /// Returns the name of the slot, or "$temp" if the slot is nameless.
  /// When creating fields in the component the symbol table will take care of renaming them to a
  /// non-conflicting name.
  virtual mlir::StringRef getSlotName() const;

  virtual void rename(llvm::StringRef);

  FrameSlotKind getKind() const;

protected:
  FrameSlot(FrameSlotKind);
  FrameSlot(FrameSlotKind, mlir::StringRef);

private:
  const FrameSlotKind kind;
  llvm::SmallString<10> name;
};

} // namespace zhl

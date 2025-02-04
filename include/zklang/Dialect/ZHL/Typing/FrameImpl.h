#pragma once

#include "zklang/Dialect/ZHL/Typing/Frame.h"
#include "zklang/Dialect/ZHL/Typing/FrameSlot.h"
#include <llvm/Support/Debug.h>

namespace zhl {
namespace detail {

template <typename Slot, typename... Args>
FrameSlot *FrameInfo::allocateSlot(Args &&...args, Frame &parent) {
  FrameSlot *slot = new Slot(std::forward<Args>(args)...);
  llvm::dbgs() << "Slot: " << slot << "\n";
  llvm::dbgs() << "Slots empty? " << slots.empty() << "\n";
  slots.push_back(*slot);
  return slot;
}

} // namespace detail
} // namespace zhl

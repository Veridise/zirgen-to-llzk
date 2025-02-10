#pragma once

#include "zklang/Dialect/ZHL/Typing/Frame.h"
#include "zklang/Dialect/ZHL/Typing/FrameSlot.h"
#include <llvm/Support/Debug.h>

namespace zhl {
namespace detail {

template <typename Slot, typename... Args> FrameSlot *FrameInfo::allocateSlot(Args &&...args) {
  FrameSlot *slot = new Slot(std::forward<Args>(args)...);
  slots.push_back(*slot);
  slot->setParent(this);
  return slot;
}

} // namespace detail
} // namespace zhl

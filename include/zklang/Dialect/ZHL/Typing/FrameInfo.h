#pragma once

#include <llvm/ADT/simple_ilist.h>

namespace zhl {

class Frame;
class FrameSlot;
namespace detail {

class FrameInfo {
public:
  ~FrameInfo();
  template <typename Slot, typename... Args> FrameSlot *allocateSlot(Args &&...args, Frame &);

  using SlotsList = llvm::simple_ilist<FrameSlot>;

  SlotsList::iterator begin();
  SlotsList::const_iterator begin() const;
  SlotsList::iterator end();
  SlotsList::const_iterator end() const;

private:
  SlotsList slots;
};

} // namespace detail

} // namespace zhl

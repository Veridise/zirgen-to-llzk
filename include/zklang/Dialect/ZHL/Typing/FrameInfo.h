#pragma once

#include <llvm/ADT/simple_ilist.h>

namespace zhl {

class Frame;
class FrameSlot;

namespace detail {

class FrameInfo {
public:
  // FrameInfo();
  ~FrameInfo();
  template <typename Slot, typename... Args> FrameSlot *allocateSlot(Args &&...args);

  using SlotsList = llvm::simple_ilist<FrameSlot>;

  SlotsList::iterator begin();
  SlotsList::const_iterator begin() const;
  SlotsList::iterator end();
  SlotsList::const_iterator end() const;

  void setParentSlot(FrameSlot *);
  FrameSlot *getParentSlot() const;

private:
  FrameSlot *parent = nullptr;
  SlotsList slots;
};

} // namespace detail

} // namespace zhl

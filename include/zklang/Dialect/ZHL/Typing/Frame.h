#pragma once

#include <llvm/ADT/simple_ilist.h>
#include <memory>
#include <zklang/Dialect/ZHL/Typing/FrameInfo.h>

namespace zhl {

class FrameSlot;
class TypeBinding;

// Represents the memory used by a component. Collects slots that contain
// components or other frames. This information is later used to generate
// field reads and writes in the LLZK structs for the operations that are
// needed in the constrain function but cannot be generated there.
class Frame {
public:
  Frame();
  Frame(const Frame &);
  Frame(Frame &&) = delete;
  Frame &operator=(const Frame &);
  Frame &operator=(Frame &&) = delete;

  template <typename Slot, typename... Args> Slot *allocateSlot(Args &&...args) {
    return static_cast<Slot *>(info->allocateSlot<Slot, Args...>(std::forward<Args>(args)...));
  }

  detail::FrameInfo::SlotsList::iterator begin();
  detail::FrameInfo::SlotsList::const_iterator begin() const;
  detail::FrameInfo::SlotsList::iterator end();
  detail::FrameInfo::SlotsList::const_iterator end() const;

  // The frame information forms a tree of slots.
  // Thus, each frame may be contained inside a slot of a frame in an upper level of the hierarchy.
  // Once set, overwriting the parent is considered a bug.
  void setParentSlot(FrameSlot *);
  FrameSlot *getParentSlot() const;

private:
  // A pointer to the unique information about the frame
  std::shared_ptr<detail::FrameInfo> info;
};

} // namespace zhl

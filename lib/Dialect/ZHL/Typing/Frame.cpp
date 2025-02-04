#include "zklang/Dialect/ZHL/Typing/Frame.h"
#include "zklang/Dialect/ZHL/Typing/InnerFrame.h"
#include <memory>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/ArrayFrame.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/FrameInfo.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

namespace zhl {

Frame::Frame() : info(std::make_shared<detail::FrameInfo>()) {}

Frame::Frame(const Frame &other) : info(other.info) {}

Frame &Frame::operator=(const Frame &other) {
  info = other.info;
  return *this;
}

detail::FrameInfo::SlotsList::iterator Frame::begin() { return info->begin(); }
detail::FrameInfo::SlotsList::const_iterator Frame::begin() const { return info->begin(); }
detail::FrameInfo::SlotsList::iterator Frame::end() { return info->end(); }
detail::FrameInfo::SlotsList::const_iterator Frame::end() const { return info->end(); }

FrameSlot::FrameSlot(FrameSlotKind slotKind) : FrameSlot(slotKind, "$temp") {}

FrameSlot::FrameSlot(FrameSlotKind slotKind, mlir::StringRef slotName)
    : kind(slotKind), name(slotName) {}

void FrameSlot::rename(mlir::StringRef newName) { name = newName; }

mlir::StringRef FrameSlot::getSlotName() const { return name; }

FrameSlot::FrameSlotKind FrameSlot::getKind() const { return kind; }

InnerFrame::InnerFrame() : FrameSlot(FS_Frame, "$inner") {}

Frame &InnerFrame::getFrame() { return innerFrame; }

bool InnerFrame::classof(const FrameSlot *S) { return S->getKind() == FS_Frame; }

ArrayFrame::ArrayFrame() : FrameSlot(FS_Array, "$array") {}

bool ArrayFrame::classof(const FrameSlot *S) { return S->getKind() == FS_Array; }

Frame &ArrayFrame::getFrame() { return innerFrame; }

namespace detail {

FrameInfo::~FrameInfo() { slots.clearAndDispose(std::default_delete<FrameSlot>()); }

FrameInfo::SlotsList::iterator FrameInfo::begin() { return slots.begin(); }
FrameInfo::SlotsList::const_iterator FrameInfo::begin() const { return slots.begin(); }
FrameInfo::SlotsList::iterator FrameInfo::end() { return slots.end(); }
FrameInfo::SlotsList::const_iterator FrameInfo::end() const { return slots.end(); }

} // namespace detail

ComponentSlot::ComponentSlot(TypeBinding &type) : FrameSlot(FS_Component), binding(type) {
  // Mark both the input argument and the member since we make a copy of the binding
  binding.markSlot(this);
  type.markSlot(this);
}

ComponentSlot::ComponentSlot(TypeBinding &type, mlir::StringRef name)
    : FrameSlot(FS_Component, name), binding(type) {
  // Mark both the input argument and the member since we make a copy of the binding
  binding.markSlot(this);
  type.markSlot(this);
}

bool ComponentSlot::classof(const FrameSlot *S) { return S->getKind() == FS_Component; }

} // namespace zhl

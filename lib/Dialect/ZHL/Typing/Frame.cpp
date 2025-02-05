#include "zklang/Dialect/ZHL/Typing/Frame.h"
#include "zklang/Dialect/ZHL/Typing/InnerFrame.h"
#include <memory>
#include <mlir/IR/Value.h>
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

void Frame::setParentSlot(FrameSlot *slot) { info->setParentSlot(slot); }
FrameSlot *Frame::getParentSlot() const { return info->getParentSlot(); }

FrameSlot::FrameSlot(FrameSlotKind slotKind) : FrameSlot(slotKind, "$temp") {}

FrameSlot::FrameSlot(FrameSlotKind slotKind, mlir::StringRef slotName)
    : kind(slotKind), name(slotName) {}

void FrameSlot::rename(mlir::StringRef newName) { name = newName; }

mlir::StringRef FrameSlot::getSlotName() const { return name; }

FrameSlot::FrameSlotKind FrameSlot::getKind() const { return kind; }

void FrameSlot::setParent(const detail::FrameInfo *frame) { parentFrame = frame; }

FrameSlot *FrameSlot::getParentSlot() const {
  if (!parentFrame) {
    return nullptr;
  }
  return parentFrame->getParentSlot();
}

const detail::FrameInfo *FrameSlot::getParent() const { return parentFrame; }

InnerFrame::InnerFrame() : FrameSlot(FS_Frame, "$inner"), innerFrame{} {
  innerFrame.setParentSlot(this);
}

Frame &InnerFrame::getFrame() { return innerFrame; }

bool InnerFrame::classof(const FrameSlot *S) { return S->getKind() == FS_Frame; }

ArrayFrame::ArrayFrame() : FrameSlot(FS_Array, "$array"), iv(nullptr), innerFrame{} {
  innerFrame.setParentSlot(this);
}

bool ArrayFrame::classof(const FrameSlot *S) { return S->getKind() == FS_Array; }

Frame &ArrayFrame::getFrame() { return innerFrame; }

void ArrayFrame::setInductionVar(mlir::Value val) { iv = val; }

mlir::Value ArrayFrame::getInductionVar() const {
  assert(iv && "induction var retrieved before having been set");
  return iv;
}

namespace detail {

// FrameInfo::FrameInfo() {
//   assert(slots.end()->isKnownSentinel());
//   // slots.end()->setParent(this);
// }
FrameInfo::~FrameInfo() { slots.clearAndDispose(std::default_delete<FrameSlot>()); }

FrameInfo::SlotsList::iterator FrameInfo::begin() { return slots.begin(); }
FrameInfo::SlotsList::const_iterator FrameInfo::begin() const { return slots.begin(); }
FrameInfo::SlotsList::iterator FrameInfo::end() { return slots.end(); }
FrameInfo::SlotsList::const_iterator FrameInfo::end() const { return slots.end(); }

void FrameInfo::setParentSlot(FrameSlot *slot) {
  assert(!parent && "cannot set the parent slot of a frame twice");
  parent = slot;
}
FrameSlot *FrameInfo::getParentSlot() const { return parent; }

} // namespace detail

ComponentSlot::ComponentSlot(const TypeBindings &bindings, TypeBinding &type)
    : FrameSlot(FS_Component), binding(type), bindingsCtx(&bindings) {
  // Mark both the input argument and the member since we make a copy of the binding
  binding.markSlot(this);
  type.markSlot(this);
}

ComponentSlot::ComponentSlot(const TypeBindings &bindings, TypeBinding &type, mlir::StringRef name)
    : FrameSlot(FS_Component, name), binding(type), bindingsCtx(&bindings) {
  // Mark both the input argument and the member since we make a copy of the binding
  binding.markSlot(this);
  type.markSlot(this);
}

bool ComponentSlot::classof(const FrameSlot *S) { return S->getKind() == FS_Component; }

template <typename T>
T traverseNestedArrayFramesT(
    const FrameSlot *slot, const T &t, std::function<T(const ArrayFrame *, const T &)> onArrayFrame
) {
  if (!slot) {
    return t;
  }
  if (auto arrayFrame = mlir::dyn_cast<ArrayFrame>(slot)) {
    /// Make a copy of the array just created to make sure it survives until the recursive function
    /// finishes
    auto arrayT = onArrayFrame(arrayFrame, t);
    return traverseNestedArrayFramesT(slot->getParentSlot(), arrayT, onArrayFrame);
  }
  return traverseNestedArrayFramesT(slot->getParentSlot(), t, onArrayFrame);
}

void traverseNestedArrayFrames(
    const FrameSlot *slot, std::function<void(const ArrayFrame *)> onArrayFrame
) {
  size_t dummy = 0;
  traverseNestedArrayFramesT<size_t>(slot, dummy, [&](const ArrayFrame *frame, auto &) {
    onArrayFrame(frame);
    return 0;
  });
}

TypeBinding ComponentSlot::getBinding() const {
  return traverseNestedArrayFramesT<TypeBinding>(
      this, binding,
      [&](const ArrayFrame *, const TypeBinding &b) {
    return bindingsCtx->UnkArray(b, b.getLocation());
  }
  );
}

mlir::ValueRange ComponentSlot::collectIVs() const {
  mlir::SmallVector<mlir::Value> vec;
  traverseNestedArrayFrames(this, [&](const ArrayFrame *frame) {
    vec.insert(vec.begin(), frame->getInductionVar());
  });
  return mlir::ValueRange{vec};
}

} // namespace zhl

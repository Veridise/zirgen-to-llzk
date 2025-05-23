//===- Frame.cpp - Memory frames for components -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <zklang/Dialect/ZHL/Typing/Frame.h>

#include <memory>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/ArrayFrame.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/FrameInfo.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/InnerFrame.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

#define DEBUG_TYPE "zhl-frame"

namespace zhl {

Frame::Frame() : info(std::make_shared<detail::FrameInfo>()) {}

Frame::Frame(const Frame &other) : info(other.info) {}

Frame::Frame(Frame &&other) : info(std::move(other.info)) {}

Frame &Frame::operator=(const Frame &other) {
  info = other.info;
  return *this;
}

Frame &Frame::operator=(Frame &&other) {
  if (this != &other) {
    info = std::move(other.info);
  }
  return *this;
}

void Frame::print(llvm::raw_ostream &os) const { info->print(os); }

Frame::iterator Frame::begin() { return info->begin(); }
Frame::const_iterator Frame::begin() const { return info->begin(); }
Frame::iterator Frame::end() { return info->end(); }
Frame::const_iterator Frame::end() const { return info->end(); }

void Frame::setParentSlot(FrameSlot *slot) { info->setParentSlot(slot); }
FrameSlot *Frame::getParentSlot() const { return info->getParentSlot(); }

// FrameSlot

FrameSlot::FrameSlot(FrameSlotKind slotKind) : FrameSlot(slotKind, "") {}

FrameSlot::FrameSlot(FrameSlotKind slotKind, mlir::StringRef slotName)
    : kind(slotKind), name(slotName) {}

void FrameSlot::rename(mlir::StringRef newName) { name = newName; }

mlir::StringRef FrameSlot::getSlotName() const {
  return isTemporary() ? defaultNameForTemporaries() : mlir::StringRef(name);
}

mlir::StringRef FrameSlot::defaultNameForTemporaries() const { return "$temp"; }

bool FrameSlot::isTemporary() const { return name.empty(); }

FrameSlot::FrameSlotKind FrameSlot::getKind() const { return kind; }

void FrameSlot::setParent(const detail::FrameInfo *frame) { parentFrame = frame; }

FrameSlot *FrameSlot::getParentSlot() const {
  if (!parentFrame) {
    return nullptr;
  }
  return parentFrame->getParentSlot();
}

bool FrameSlot::belongsTo(const Frame &frame) const {
  // If the given frame points to the same as my parent then the slot belongs to the frame.
  if (frame.info.get() == parentFrame) {
    return true;
  }

  // If the frame has a parent slot, then the slot belongs to the frame if the parent slot belongs
  // to the frame.
  if (auto *parentSlot = getParentSlot()) {
    return parentSlot->belongsTo(frame);
  }
  // Otherwise the slot does not belong to the frame.
  return false;
}

const detail::FrameInfo *FrameSlot::getParent() const { return parentFrame; }

void FrameSlot::print(llvm::raw_ostream &os) const {
  os << "slot { parent = " << parentFrame << ", name = " << name << " }";
}

InnerFrame::InnerFrame(const TypeBindings &bindings)
    : ComponentSlot(FS_Frame, bindings, bindings.Component()), innerFrame{} {
  innerFrame.setParentSlot(this);
}

mlir::StringRef InnerFrame::defaultNameForTemporaries() const { return "$inner"; }

Frame &InnerFrame::getFrame() { return innerFrame; }

bool InnerFrame::classof(const FrameSlot *S) { return S->getKind() == FS_Frame; }

void InnerFrame::print(llvm::raw_ostream &os) const {
  ComponentSlot::print(os);
  os << " + { frame = ";
  innerFrame.print(os);
  os << " }";
}

ArrayFrame::ArrayFrame(const TypeBindings &bindings)
    : ComponentSlot(FS_Array, bindings, bindings.Component()), iv(nullptr), innerFrame{},
      size(bindings.UnkConst()) {
  innerFrame.setParentSlot(this);
}

bool ArrayFrame::classof(const FrameSlot *S) { return S->getKind() == FS_Array; }

mlir::StringRef ArrayFrame::defaultNameForTemporaries() const { return "$array"; }

Frame &ArrayFrame::getFrame() { return innerFrame; }

void ArrayFrame::setInductionVar(mlir::Value val) { iv = val; }

mlir::Value ArrayFrame::getInductionVar() const {
  assert(iv && "induction var retrieved before having been set");
  return iv;
}

void ArrayFrame::print(llvm::raw_ostream &os) const {
  ComponentSlot::print(os);
  os << " + { iv = " << iv << ", frame = ";
  innerFrame.print(os);
  os << " }";
}

namespace detail {

FrameInfo::~FrameInfo() {
  assert(slots.size() == nCreatedSlots);
  slots.clearAndDispose(std::default_delete<FrameSlot>());
}

FrameInfo::SlotsList::iterator FrameInfo::begin() { return slots.begin(); }
FrameInfo::SlotsList::const_iterator FrameInfo::begin() const { return slots.begin(); }
FrameInfo::SlotsList::iterator FrameInfo::end() { return slots.end(); }
FrameInfo::SlotsList::const_iterator FrameInfo::end() const { return slots.end(); }

void FrameInfo::setParentSlot(FrameSlot *slot) {
  assert(!parent && "cannot set the parent slot of a frame twice");
  parent = slot;
}
FrameSlot *FrameInfo::getParentSlot() const { return parent; }

void FrameInfo::print(llvm::raw_ostream &os) const {
  os << "frame " << this << " [";
  llvm::interleaveComma(slots, os, [&](auto &slot) { slot.print(os); });
  os << "]";
}

} // namespace detail

ComponentSlot::ComponentSlot(const TypeBindings &Bindings, TypeBinding &Type)
    : FrameSlot(FS_Component), binding(Type), bindingsCtx(&Bindings) {
  // Mark both the input argument and the member since we make a copy of the binding
  binding.markSlot(this);
  Type.markSlot(this);
}

ComponentSlot::ComponentSlot(const TypeBindings &Bindings, TypeBinding &Type, mlir::StringRef Name)
    : FrameSlot(FS_Component, Name), binding(Type), bindingsCtx(&Bindings) {
  // Mark both the input argument and the member since we make a copy of the binding
  binding.markSlot(this);
  Type.markSlot(this);
}

ComponentSlot::ComponentSlot(
    FrameSlotKind Kind, const TypeBindings &Bindings, const TypeBinding &Type
)
    : FrameSlot(Kind), binding(Type), bindingsCtx(&Bindings) {
  binding.markSlot(this);
}

bool ComponentSlot::classof(const FrameSlot *S) {
  return S->getKind() >= FS_Component && S->getKind() < FS_ComponentEnd;
}

void ComponentSlot::print(llvm::raw_ostream &os) const {
  FrameSlot::print(os);
  os << " + { binding = " << binding << " }";
}

template <typename T>
T traverseNestedArrayFramesT(
    const FrameSlot *root, const FrameSlot *slot, const T &t,
    std::function<T(const ArrayFrame &, const T &)> onArrayFrame
) {
  if (!slot) {
    return t;
  }
  if (root != slot) {
    if (auto arrayFrame = mlir::dyn_cast<ArrayFrame>(slot)) {
      /// Make a copy of the array just created to make sure it survives until the recursive
      /// function finishes
      auto arrayT = onArrayFrame(*arrayFrame, t);
      return traverseNestedArrayFramesT(root, slot->getParentSlot(), arrayT, onArrayFrame);
    }
  }
  return traverseNestedArrayFramesT(root, slot->getParentSlot(), t, onArrayFrame);
}

static void traverseNestedArrayFrames(
    const FrameSlot *root, const FrameSlot *slot,
    std::function<void(const ArrayFrame &)> onArrayFrame
) {
  size_t dummy = 0;
  traverseNestedArrayFramesT<size_t>(root, slot, dummy, [&](const ArrayFrame &frame, auto &) {
    onArrayFrame(frame);
    return 0;
  });
}

void ComponentSlot::setBinding(TypeBinding &newBinding) {
  newBinding.markSlot(this);
  binding = newBinding;
}

TypeBinding ComponentSlot::getBinding() const {
  return traverseNestedArrayFramesT<TypeBinding>(
      this, this, binding,
      [&](const ArrayFrame &frame, const TypeBinding &b) {
    return bindingsCtx->Array(b, frame.getSize(), b.getLocation());
  }
  );
}

mlir::SmallVector<mlir::Value> ComponentSlot::collectIVs() const {
  mlir::SmallVector<mlir::Value> vec;
  traverseNestedArrayFrames(this, this, [&](const ArrayFrame &frame) {
    vec.insert(vec.begin(), frame.getInductionVar());
  });
  return vec;
}

bool ComponentSlot::contains(const TypeBinding &other) const { return binding == other; }

void ComponentSlot::editInnerBinding(llvm::function_ref<void(TypeBinding &)> edit) {
  edit(binding);
}

} // namespace zhl

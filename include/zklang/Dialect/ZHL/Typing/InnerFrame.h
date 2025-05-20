//===- InnerFrame.h - A frame inside another --------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a class that represents a frame that is contained within
// another. Used to represent blocks in the zirgen DSL.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>

namespace zhl {

/// Represents a Frame inside another
class InnerFrame : public ComponentSlot {
public:
  InnerFrame(const TypeBindings &);
  Frame &getFrame();

  static bool classof(const FrameSlot *);

  void print(llvm::raw_ostream &) const override;

  llvm::hash_code hash() const override;

protected:
  mlir::StringRef defaultNameForTemporaries() const override;

private:
  Frame innerFrame;
};

} // namespace zhl

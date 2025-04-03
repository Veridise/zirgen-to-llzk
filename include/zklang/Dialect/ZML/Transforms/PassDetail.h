//===- PassDetail.h - Pass inner details ===---------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the internal definitions of the passes defined in
// Passes.td in the same directory.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Pass/Pass.h>                // IWYU pragma: keep
#include <zklang/Dialect/ZML/IR/Dialect.h> // IWYU pragma: keep
#include <zklang/Dialect/ZML/IR/Ops.h>     // IWYU pragma: keep

namespace zml {

#define GEN_PASS_CLASSES
#include <zklang/Dialect/ZML/Transforms/Passes.h.inc>

} // namespace zml

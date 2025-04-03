//===- PassDetail.h - Pass details ------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the tablegen generated header file for the passes
// defined in Passes.td in the same directory.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h> // IWYU pragma: keep
#include <zirgen/Dialect/ZHL/IR/ZHL.h>

namespace zhl {

#define GEN_PASS_CLASSES
#include <zklang/Dialect/ZHL/Typing/Passes.h.inc>

} // namespace zhl

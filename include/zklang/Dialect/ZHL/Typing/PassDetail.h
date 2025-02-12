// Copyright 2024 Veridise, Inc.

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h> // IWYU pragma: keep
#include <zirgen/Dialect/ZHL/IR/ZHL.h>

namespace zhl {

#define GEN_PASS_CLASSES
#include <zklang/Dialect/ZHL/Typing/Passes.h.inc>

} // namespace zhl

// Copyright 2024 Veridise, Inc.

#pragma once

#include <llzk/Dialect/LLZK/IR/Dialect.h>  // IWYU pragma: keep
#include <llzk/Dialect/LLZK/IR/Ops.h>      // IWYU pragma: keep
#include <mlir/Pass/Pass.h>                // IWYU pragma: keep
#include <zirgen/Dialect/ZHL/IR/ZHL.h>     // IWYU pragma: keep
#include <zklang/Dialect/ZML/IR/Dialect.h> // IWYU pragma: keep
#include <zklang/Dialect/ZML/IR/Ops.h>     // IWYU pragma: keep

namespace zklang {

#define GEN_PASS_CLASSES
#include <zklang/Passes/Passes.h.inc>

} // namespace zklang

// Copyright 2024 Veridise, Inc.

#pragma once

#include <mlir/Pass/Pass.h>                // IWYU pragma: keep
#include <zklang/Dialect/ZML/IR/Dialect.h> // IWYU pragma: keep
#include <zklang/Dialect/ZML/IR/Ops.h>     // IWYU pragma: keep

namespace zml {

#define GEN_PASS_CLASSES
#include <zklang/Dialect/ZML/Transforms/Passes.h.inc>

} // namespace zml

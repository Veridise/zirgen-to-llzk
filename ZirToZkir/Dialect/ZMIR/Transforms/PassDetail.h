// Copyright 2024 Veridise, Inc.

#pragma once

#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h" // IWYU pragma: keep
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"     // IWYU pragma: keep
#include "mlir/Pass/Pass.h"                    // IWYU pragma: keep

namespace zkc::Zmir {

#define GEN_PASS_CLASSES
#include "ZirToZkir/Dialect/ZMIR/Transforms/Passes.inc.h"

} // namespace zkc::Zmir

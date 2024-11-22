// Copyright 2024 Veridise, Inc.

#pragma once

#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h" // IWYU pragma: keep
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"     // IWYU pragma: keep
#include "mlir/Pass/Pass.h"                    // IWYU pragma: keep
#include "zirgen/Dialect/ZHL/IR/ZHL.h"         // IWYU pragma: keep
#include "zkir/Dialect/ZKIR/IR/Dialect.h"      // IWYU pragma: keep

namespace zkc {

#define GEN_PASS_CLASSES
#include "ZirToZkir/Passes/Passes.inc.h"

} // namespace zkc

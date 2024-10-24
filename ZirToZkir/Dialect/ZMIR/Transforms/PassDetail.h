// Copyright 2024 Veridise, Inc.

#pragma once

#include "mlir/Pass/Pass.h"

#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h"
#include "ZirToZkir/Dialect/ZMIR/Transforms/Passes.h"

namespace zkc {

#define GEN_PASS_CLASSES
#include "ZirToZkir/Dialect/ZMIR/Transforms/Passes.h.inc"

} // namespace zkc

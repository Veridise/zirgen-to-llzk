// Copyright 2024 Veridise, Inc.

#pragma once

/*#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h" // IWYU pragma: keep*/
/*#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"     // IWYU pragma: keep*/
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h" // IWYU pragma: keep
#include "zirgen/Dialect/ZHL/IR/ZHL.h"

namespace zhl {

#define GEN_PASS_CLASSES
#include "ZirToZkir/Dialect/ZHL/Typing/Passes.inc.h"

} // namespace zhl

//===- Driver.h - Zirgen driver ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the main entrypoint for the zirgen frontend.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Support/LogicalResult.h>

namespace zklang {

/// Zirgen frontend driver
mlir::LogicalResult zirDriver(int &, char **&);

} // namespace zklang

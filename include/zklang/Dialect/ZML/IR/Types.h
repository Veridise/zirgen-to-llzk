//===- Types.h - ZML types --------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the ZML type declarations generated from Types.td.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/MemorySlotInterfaces.h>
#include <zklang/Dialect/ZML/IR/OpInterfaces.h>
#include <zklang/Dialect/ZML/IR/TypeInterfaces.h>

namespace zml {

bool isValidZMLType(mlir::Type);

}

// forward-declare ops
#define GET_OP_FWD_DEFINES
#include <zklang/Dialect/ZML/IR/Ops.h.inc>

// Include TableGen'd declarations
#define GET_TYPEDEF_CLASSES
#include <zklang/Dialect/ZML/IR/Types.h.inc>

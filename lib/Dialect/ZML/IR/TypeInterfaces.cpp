//===- TypeInterfaces.cpp - ZML type interfaces -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <functional>
#include <llvm/ADT/SmallVectorExtras.h>
#include <zklang/Dialect/ZML/IR/Types.h>

#include <algorithm>
#include <iterator>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/TypeInterfaces.h>

// namespace zml {
// bool defaultSubtypeOfImpl(ComponentLike, ComponentLike);
// }

// TableGen'd implementation files
#define GET_OP_CLASSES
#include <zklang/Dialect/ZML/IR/TypeInterfaces.cpp.inc>

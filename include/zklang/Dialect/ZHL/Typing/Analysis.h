//===- Analysis.h - Type Analysis entrypoint --------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes an analysis interface for accessing type analysis results.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Value.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

namespace zhl {

class ZIRTypeAnalysis {
public:
  ZIRTypeAnalysis(mlir::Operation *op, mlir::AnalysisManager &am);
  ~ZIRTypeAnalysis();

  const mlir::FailureOr<TypeBinding> &getType(mlir::Operation *op) const;

  const mlir::FailureOr<TypeBinding> &getType(mlir::Value value) const;

  mlir::FailureOr<TypeBinding> getType(mlir::StringRef name) const;

  mlir::LogicalResult addType(mlir::Value value, const TypeBinding &binding);

  const TypeBindings &getBindings() const;

  void dump() const;

  void print(llvm::raw_ostream &os) const;

  mlir::ArrayRef<TypeBinding *> getClosures();

  operator mlir::LogicalResult() const;

  void emitRemarks() const;

  class Impl;

private:
  std::unique_ptr<Impl> impl;
};

} // namespace zhl

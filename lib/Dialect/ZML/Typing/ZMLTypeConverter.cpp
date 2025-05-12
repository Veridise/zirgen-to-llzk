//===- ZMLTypeConverter.cpp - ZML type conversion ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <mlir/IR/BuiltinOps.h>
#include <optional>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/Types.h>
#include <zklang/Dialect/ZML/Typing/ZMLTypeConverter.h>

using namespace zml;
using namespace zirgen;

ZMLTypeConverter::ZMLTypeConverter() {
  addConversion([](mlir::Type t) { return t; });

  addSourceMaterialization(
      [](mlir::OpBuilder &builder, Zhl::ExprType type, mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
    return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
  }
  );
  addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
    if (inputs.size() != 1) {
      return std::nullopt;
    }

    return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
  }
  );
  addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
    if (!mlir::isa<zml::ComponentLike>(type)) {
      return std::nullopt;
    }
    if (auto compTyp = mlir::dyn_cast<zml::ComponentLike>(inputs[0].getType())) {
      if (compTyp.getName().getValue() != "Component") {
        return builder.create<zml::SuperCoerceOp>(loc, type, inputs[0]);
      }
    }
    return std::nullopt;
  }
  );

  addArgumentMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
         mlir::Location loc) -> std::optional<mlir::Value> {
    return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
  }
  );
}

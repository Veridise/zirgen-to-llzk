//===- Builder.h - ZML Component builder ------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a builder class for creating instances for ComponentOp
// operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cassert>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <vector>
#include <zklang/Dialect/ZML/IR/Ops.h>

namespace zml {

using Identifier = mlir::SmallString<10>;

class ComponentBuilder {
private:
  struct Field {
    Identifier name;
    mlir::Type type;
    std::optional<mlir::Location> loc;
  };
  struct Ctx;

  class BodySrc {
  public:
    virtual ~BodySrc();
    virtual void set(ComponentOp op, Ctx &ctx, mlir::OpBuilder &builder) const = 0;
  };

  struct Ctx {
    std::optional<mlir::Location> loc;
    Identifier compName;
    mlir::SmallVector<Field> fields;
    mlir::SmallVector<mlir::NamedAttribute> compAttrs;
    mlir::SmallVector<Identifier> typeParams;
    mlir::FunctionType constructorType = nullptr;
    mlir::SmallVector<mlir::Location> argLocs;
    std::unique_ptr<BodySrc> body;
    std::function<void(ComponentOp)> deferCb = nullptr;
    bool isBuiltin = false;
    bool isClosure = false;
    bool forceSetGeneric = false;
    bool usesBackVariables = false;

    bool isGeneric();
    // Builds a bare component op
    ComponentOp buildBare(mlir::OpBuilder &builder);

    void checkRequirements();
    void checkBareRequirements();

    void addFields(mlir::OpBuilder &builder);

    void addBody(ComponentOp op, mlir::OpBuilder &builder);

    mlir::SmallVector<mlir::NamedAttribute> funcBodyAttrs(mlir::OpBuilder &builder);

    mlir::SmallVector<mlir::NamedAttribute> builtinAttrs(mlir::OpBuilder &builder);
  };

  class TakeRegion : public BodySrc {
  public:
    explicit TakeRegion(mlir::Region *body);

    void set(ComponentOp op, Ctx &ctx, mlir::OpBuilder &builder) const override;

  private:
    mlir::Region *body = nullptr;
  };

  class FillBody : public BodySrc {
  public:
    FillBody(
        mlir::ArrayRef<mlir::Type> argTypes, mlir::ArrayRef<mlir::Type> results,
        std::function<void(mlir::ValueRange, mlir::OpBuilder &)> delegate
    );

    void set(ComponentOp op, Ctx &ctx, mlir::OpBuilder &builder) const override;

  private:
    mlir::SmallVector<mlir::Type, 1> argTypes, results;
    std::function<void(mlir::ValueRange, mlir::OpBuilder &)> delegate;
  };

  Ctx ctx;

public:
  ComponentBuilder &forceGeneric();

  ComponentBuilder &isBuiltin();

  ComponentBuilder &isClosure();

  ComponentBuilder &usesBackVariables();

  ComponentBuilder &takeRegion(mlir::Region *region);

  ComponentBuilder &fillBody(
      mlir::ArrayRef<mlir::Type> argTypes, mlir::ArrayRef<mlir::Type> results,
      std::function<void(mlir::ValueRange, mlir::OpBuilder &)> fn
  );

  ComponentBuilder &
  constructor(mlir::FunctionType constructorType, mlir::ArrayRef<mlir::Location> argLocs);

  ComponentBuilder &location(mlir::Location loc);

  ComponentBuilder &name(mlir::StringRef name);

  ComponentBuilder &field(mlir::StringRef name, mlir::Type type, mlir::Location loc);

  ComponentBuilder &field(mlir::StringRef name, mlir::Type type);

  ComponentBuilder &attrs(mlir::ArrayRef<mlir::NamedAttribute> attrs);

  ComponentBuilder &typeParam(mlir::StringRef param);

  ComponentBuilder &typeParams(mlir::ArrayRef<std::string> params);

  /// Stores a callback that gets executed right after the component op is created but before it is
  /// filled with the rest of the configuration. Allows configuring the builder with information
  /// that depends on the ComponentOp operation having already been created, such as inserting the
  /// op in a symbol table and getting potentially renamed.
  ComponentBuilder &defer(std::function<void(ComponentOp)>);

  ComponentOp build(mlir::OpBuilder &builder);
};

} // namespace zml

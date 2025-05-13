//===- Builder.h - LLZK Component builder ------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a builder class for creating instances for struct.def
// operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZML/IR/Ops.h>

namespace mlir {
class TypeConverter;
}

namespace llzk {

using Identifier = mlir::SmallString<10>;

class ComponentBuilder {
public:
  using FillFn =
      std::function<void(mlir::ValueRange, mlir::OpBuilder &, const mlir::TypeConverter &)>;

  using type = component::StructDefOp;

private:
  struct Field {
    Identifier name;
    mlir::Type type;
    std::optional<mlir::Location> loc;
    bool column : 1;
  };
  struct Ctx;

  class BodySrc {
  public:
    virtual ~BodySrc();
    virtual void
    set(type op, Ctx &ctx, mlir::OpBuilder &builder, const mlir::TypeConverter &) const = 0;
  };

  struct Ctx {
    std::optional<mlir::Location> loc;
    Identifier compName;
    mlir::SmallVector<Field> fields;
    mlir::SmallVector<mlir::NamedAttribute> compAttrs;
    mlir::SmallVector<Identifier> typeParams;
    mlir::FunctionType constructorType = nullptr;
    mlir::FunctionType constrainFnType = nullptr;
    std::optional<mlir::SmallVector<mlir::Location>> argLocs;
    std::unique_ptr<BodySrc> body;
    std::function<void(type)> deferCb = nullptr;
    bool isBuiltin = false;
    bool isClosure = false;
    bool forceSetGeneric = false;
    bool usesBackVariables = false;

    bool isGeneric();
    // Builds a bare component op
    type buildBare(mlir::OpBuilder &builder);

    void checkRequirements();
    void checkBareRequirements();

    void addFields(mlir::OpBuilder &builder);

    void addBody(type op, mlir::OpBuilder &builder, const mlir::TypeConverter &);

    mlir::SmallVector<mlir::NamedAttribute> funcBodyAttrs(mlir::OpBuilder &builder);

    mlir::SmallVector<mlir::NamedAttribute> builtinAttrs(mlir::OpBuilder &builder);
  };

  class TakeRegion : public BodySrc {
  public:
    explicit TakeRegion(mlir::Region *body);

    void
    set(type op, Ctx &ctx, mlir::OpBuilder &builder, const mlir::TypeConverter &) const override;

  private:
    mlir::Region *body = nullptr;
  };

  class FillBody : public BodySrc {
  public:
    FillBody(
        mlir::ArrayRef<mlir::Type> argTypes, mlir::ArrayRef<mlir::Type> results, FillFn compute,
        FillFn constrain = nullptr
    );

    void
    set(type op, Ctx &ctx, mlir::OpBuilder &builder, const mlir::TypeConverter &) const override;

  private:
    mlir::SmallVector<mlir::Type, 1> argTypes, results;
    FillFn fillCompute, fillConstrain;
  };

  Ctx ctx;

public:
  ComponentBuilder &forceGeneric();

  ComponentBuilder &isBuiltin();

  ComponentBuilder &isClosure();

  ComponentBuilder &usesBackVariables();

  ComponentBuilder &takeRegion(mlir::Region *region);

  ComponentBuilder &fillBody(
      mlir::ArrayRef<mlir::Type> argTypes, mlir::ArrayRef<mlir::Type> results, FillFn compute,
      FillFn constrain = nullptr
  );

  ComponentBuilder &
  constructor(mlir::FunctionType constructorType, mlir::ArrayRef<mlir::Location> argLocs);

  ComponentBuilder &constructorLocs(mlir::ArrayRef<mlir::Location> argLocs);

  ComponentBuilder &location(mlir::Location loc);

  ComponentBuilder &name(mlir::StringRef name);

  ComponentBuilder &
  field(mlir::StringRef name, mlir::Type type, mlir::Location loc, bool isColumn = false);

  ComponentBuilder &field(mlir::StringRef name, mlir::Type type, bool isColumn = false);

  ComponentBuilder &attrs(mlir::ArrayRef<mlir::NamedAttribute> attrs);

  ComponentBuilder &typeParam(mlir::StringRef param);

  ComponentBuilder &typeParams(mlir::ArrayRef<std::string> params);

  /// Stores a callback that gets executed right after the component op is created but before it is
  /// filled with the rest of the configuration. Allows configuring the builder with information
  /// that depends on the ComponentOp operation having already been created, such as inserting the
  /// op in a symbol table and getting potentially renamed.
  ComponentBuilder &defer(std::function<void(type)>);

  type build(mlir::OpBuilder &builder, const mlir::TypeConverter &tc);
};

} // namespace llzk

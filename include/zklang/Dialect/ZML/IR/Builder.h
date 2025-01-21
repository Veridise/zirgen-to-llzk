#pragma once

#include "zklang/Dialect/ZML/IR/Ops.h"
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

namespace zkc::Zmir {

class ComponentBuilder {
private:
  struct Field {
    std::string_view name;
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
    std::string_view compName;
    std::vector<Field> fields;
    mlir::ArrayRef<mlir::NamedAttribute> compAttrs;
    std::vector<mlir::StringRef> typeParams;
    mlir::FunctionType constructorType = nullptr;
    std::vector<mlir::Location> argLocs;
    std::unique_ptr<BodySrc> body;
    bool isBuiltin = false;
    bool forceSetGeneric = false;

    bool isGeneric();
    // Builds a bare component op
    ComponentOp buildBare(mlir::OpBuilder &builder);

    void checkRequirements();

    void addFields(mlir::OpBuilder &builder);

    void addBody(ComponentOp op, mlir::OpBuilder &builder);

    std::vector<mlir::NamedAttribute> funcBodyAttrs(mlir::OpBuilder &builder);

    std::vector<mlir::NamedAttribute> builtinAttrs(mlir::OpBuilder &builder);
  };

  class TakeRegion : public BodySrc {
  public:
    explicit TakeRegion(mlir::Region *body);

    void set(ComponentOp op, Ctx &ctx, mlir::OpBuilder &builder) const override;

  private:
    void fillEpilogue(mlir::Block *block, ComponentOp op, mlir::OpBuilder &builder) const;

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
    mlir::ArrayRef<mlir::Type> argTypes, results;
    std::function<void(mlir::ValueRange, mlir::OpBuilder &)> delegate;
  };

  Ctx ctx;

public:
  ComponentBuilder &forceGeneric();

  ComponentBuilder &isBuiltin();

  ComponentBuilder &takeRegion(mlir::Region *region);

  ComponentBuilder &fillBody(
      mlir::ArrayRef<mlir::Type> argTypes, mlir::ArrayRef<mlir::Type> results,
      std::function<void(mlir::ValueRange, mlir::OpBuilder &)> fn
  );

  ComponentBuilder &
  constructor(mlir::FunctionType constructorType, std::vector<mlir::Location> argLocs);

  ComponentBuilder &location(mlir::Location loc);

  ComponentBuilder &name(std::string_view name);

  ComponentBuilder &field(std::string_view name, mlir::Type type, mlir::Location loc);

  ComponentBuilder &field(std::string_view name, mlir::Type type);

  ComponentBuilder &attrs(mlir::ArrayRef<mlir::NamedAttribute> attrs);

  ComponentBuilder &typeParam(mlir::StringRef param);

  ComponentBuilder &typeParams(mlir::ArrayRef<std::string> params);

  ComponentOp build(mlir::OpBuilder &builder);
};

} // namespace zkc::Zmir

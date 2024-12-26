#pragma once

#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
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
    virtual ~BodySrc() = default;
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

    bool isGeneric() {
      if (compAttrs.empty()) {
        return false;
      }
      return std::any_of(compAttrs.begin(), compAttrs.end(), [](auto attr) {
        return attr.getName() == "generic";
      });
    }
    // Builds a bare component op
    ComponentOp buildBare(mlir::OpBuilder &builder) {
      auto builtin = builtinAttrs(builder);
      mlir::ArrayRef<mlir::NamedAttribute> attrs =
          isBuiltin ? mlir::ArrayRef<mlir::NamedAttribute>(builtin) : compAttrs;

      if (isGeneric()) {
        return builder.create<ComponentOp>(*loc, compName, typeParams, attrs);
      } else {
        return builder.create<ComponentOp>(*loc, compName, attrs);
      }
    }

    void checkRequirements() {
      assert(body != nullptr);
      assert(loc.has_value());
      assert(!compName.empty());
      /*assert(isGeneric() != typeParams.empty());*/
      assert(isBuiltin == compAttrs.empty());
    }

    void addFields(mlir::OpBuilder &builder) {
      for (auto &field : fields) {
        builder.create<FieldDefOp>(
            field.loc.value_or(builder.getUnknownLoc()), builder.getStringAttr(field.name),
            mlir::TypeAttr::get(field.type)
        );
      }
    }

    void addBody(ComponentOp op, mlir::OpBuilder &builder) { body->set(op, *this, builder); }

    std::vector<mlir::NamedAttribute> funcBodyAttrs(mlir::OpBuilder &builder) {
      return {mlir::NamedAttribute(
          builder.getStringAttr("sym_visibility"), builder.getStringAttr("public")
      )};
    }

    std::vector<mlir::NamedAttribute> builtinAttrs(mlir::OpBuilder &builder) {
      return {mlir::NamedAttribute(builder.getStringAttr("builtin"), builder.getUnitAttr())};
    }
  };

  class TakeRegion : public BodySrc {
  public:
    explicit TakeRegion(mlir::Region *body) : body(body) {}

    void set(ComponentOp op, Ctx &ctx, mlir::OpBuilder &builder) const override {
      auto constructorAttrs = ctx.funcBodyAttrs(builder);

      auto bodyOp = builder.create<mlir::func::FuncOp>(
          *ctx.loc, op.getBodyFuncName(), ctx.constructorType, constructorAttrs
      );
      bodyOp.getRegion().takeBody(*body);

      // Create arguments for the entry block (aka region arguments)
      auto &entryBlock = bodyOp.front();
      entryBlock.addArguments(ctx.constructorType.getInputs(), ctx.argLocs);

      // Fill out epilogue
      auto *epilogue = bodyOp.addBlock();
      fillEpilogue(epilogue, op, builder);
    }

  private:
    void fillEpilogue(mlir::Block *block, ComponentOp op, mlir::OpBuilder &builder) const {
      mlir::OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPointToEnd(block);

      mlir::Location unk = builder.getUnknownLoc();
      auto self = builder.create<zkc::Zmir::GetSelfOp>(unk, op.getType());
      builder.create<mlir::func::ReturnOp>(unk, mlir::ValueRange({self}));
    }

    mlir::Region *body = nullptr;
  };

  class FillBody : public BodySrc {
  public:
    FillBody(
        mlir::ArrayRef<mlir::Type> argTypes, mlir::ArrayRef<mlir::Type> results,
        std::function<void(mlir::ValueRange, mlir::OpBuilder &)> delegate
    )
        : argTypes(argTypes), results(results), delegate(delegate) {}

    void set(ComponentOp op, Ctx &ctx, mlir::OpBuilder &builder) const override {
      ctx.constructorType = builder.getFunctionType(argTypes, results);
      std::vector<mlir::NamedAttribute> attrs = ctx.funcBodyAttrs(builder);

      auto bodyOp = builder.create<mlir::func::FuncOp>(
          builder.getUnknownLoc(), op.getBodyFuncName(), ctx.constructorType, attrs
      );
      mlir::OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPointToStart(bodyOp.addEntryBlock());
      delegate(bodyOp.getArguments(), builder);
    }

  private:
    mlir::ArrayRef<mlir::Type> argTypes, results;
    std::function<void(mlir::ValueRange, mlir::OpBuilder &)> delegate;
  };

  Ctx ctx;

public:
  ComponentBuilder &isBuiltin() {
    ctx.isBuiltin = true;
    return *this;
  }

  ComponentBuilder &takeRegion(mlir::Region *region) {
    ctx.body = std::make_unique<TakeRegion>(region);
    return *this;
  }

  ComponentBuilder &fillBody(
      mlir::ArrayRef<mlir::Type> argTypes, mlir::ArrayRef<mlir::Type> results,
      std::function<void(mlir::ValueRange, mlir::OpBuilder &)> fn
  ) {
    ctx.body = std::make_unique<FillBody>(argTypes, results, fn);
    return *this;
  }

  ComponentBuilder &
  constructor(mlir::FunctionType constructorType, std::vector<mlir::Location> argLocs) {
    ctx.constructorType = constructorType;
    ctx.argLocs = argLocs;
    return *this;
  }

  ComponentBuilder &location(mlir::Location loc) {
    ctx.loc = loc;
    return *this;
  }

  ComponentBuilder &name(std::string_view name) {
    ctx.compName = name;
    return *this;
  }

  ComponentBuilder &field(std::string_view name, mlir::Type type, mlir::Location loc) {
    ctx.fields.push_back({.name = name, .type = type, .loc = loc});
    return *this;
  }

  ComponentBuilder &field(std::string_view name, mlir::Type type) {
    ctx.fields.push_back({.name = name, .type = type, .loc = std::nullopt});
    return *this;
  }

  ComponentBuilder &attrs(mlir::ArrayRef<mlir::NamedAttribute> attrs) {
    ctx.compAttrs = attrs;
    return *this;
  }

  ComponentBuilder &typeParam(mlir::StringRef param) {
    ctx.typeParams.push_back(param);
    return *this;
  }

  ComponentBuilder &typeParams(mlir::ArrayRef<std::string> params) {
    for (auto &name : params) {
      typeParam(name);
    }
    return *this;
  }

  ComponentOp build(mlir::OpBuilder &builder) {
    if (!ctx.loc.has_value()) {
      ctx.loc = builder.getUnknownLoc();
    }
    ctx.checkRequirements();

    auto op = ctx.buildBare(builder);
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto block = builder.createBlock(&op.getRegion());
    builder.setInsertionPointToStart(block);
    ctx.addFields(builder);
    ctx.addBody(op, builder);

    ctx = Ctx();
    return op;
  }
};

} // namespace zkc::Zmir

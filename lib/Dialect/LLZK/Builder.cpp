//===- Builder.cpp - ZML Component builder ----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zklang/Dialect/LLZK/Builder.h>

#include <cassert>
#include <functional>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>

using namespace mlir;
using namespace llzk;

//===----------------------------------------------------------------------===//
// ComponentBuilder
//===----------------------------------------------------------------------===//

ComponentBuilder::type ComponentBuilder::build(OpBuilder &builder, const TypeConverter &tc) {
  if (!ctx.loc.has_value()) {
    ctx.loc = builder.getUnknownLoc();
  }
  ctx.checkBareRequirements();
  auto op = ctx.buildBare(builder);
  if (ctx.deferCb) {
    ctx.deferCb(op);
  }

  ctx.checkRequirements();

  OpBuilder::InsertionGuard guard(builder);
  auto block = builder.createBlock(&op.getRegion());
  builder.setInsertionPointToStart(block);
  ctx.addFields(builder);
  ctx.addBody(op, builder, tc);

  ctx = Ctx();
  return op;
}

ComponentBuilder &ComponentBuilder::typeParams(ArrayRef<std::string> params) {
  ctx.typeParams.insert(ctx.typeParams.end(), params.begin(), params.end());
  return *this;
}

ComponentBuilder &ComponentBuilder::typeParam(StringRef param) {
  ctx.typeParams.push_back(param);
  return *this;
}

ComponentBuilder &ComponentBuilder::attrs(ArrayRef<NamedAttribute> attrs) {
  ctx.compAttrs = llvm::map_to_vector(attrs, [](NamedAttribute attr) {
    auto *ctx = attr.getName().getContext();
    attr.setName(StringAttr::get(ctx, "zml." + attr.getName().getValue()));
    return attr;
  });
  return *this;
}

ComponentBuilder &ComponentBuilder::field(StringRef name, Type type, bool isColumn) {
  ctx.fields.push_back({.name = name, .type = type, .loc = std::nullopt, .column = isColumn});
  return *this;
}

ComponentBuilder &ComponentBuilder::field(StringRef name, Type type, Location loc, bool isColumn) {
  ctx.fields.push_back({.name = name, .type = type, .loc = loc, .column = isColumn});
  return *this;
}

ComponentBuilder &ComponentBuilder::name(StringRef name) {
  ctx.compName = name;
  return *this;
}

ComponentBuilder &ComponentBuilder::location(Location loc) {
  ctx.loc = loc;
  return *this;
}

ComponentBuilder &
ComponentBuilder::constructor(FunctionType constructorType, ArrayRef<Location> argLocs) {
  ctx.constructorType = constructorType;
  ctx.argLocs = SmallVector<Location>(argLocs);
  return *this;
}

ComponentBuilder &ComponentBuilder::fillBody(
    ArrayRef<Type> inputs, ArrayRef<Type> results, FillFn compute, FillFn constrain
) {
  ctx.body = std::make_unique<FillBody>(inputs, results, compute, constrain);
  return *this;
}

ComponentBuilder &ComponentBuilder::takeRegion(Region *region) {
  ctx.body = std::make_unique<TakeRegion>(region);
  return *this;
}

ComponentBuilder &ComponentBuilder::isBuiltin() {
  ctx.isBuiltin = true;
  ctx.usesBackVariables = false; // Force set it to false
  return *this;
}

ComponentBuilder &ComponentBuilder::usesBackVariables() {
  ctx.usesBackVariables = true;
  return *this;
}

ComponentBuilder &ComponentBuilder::isClosure() {
  ctx.isClosure = true;
  return *this;
}

ComponentBuilder &ComponentBuilder::forceGeneric() {
  ctx.forceSetGeneric = true;
  return *this;
}

ComponentBuilder &ComponentBuilder::defer(std::function<void(type)> cb) {
  ctx.deferCb = cb;
  return *this;
}

//===----------------------------------------------------------------------===//
// ComponentBuilder::Ctx
//===----------------------------------------------------------------------===//

bool ComponentBuilder::Ctx::isGeneric() {
  if (forceSetGeneric) {
    return true;
  }
  if (compAttrs.empty()) {
    return false;
  }
  return std::any_of(compAttrs.begin(), compAttrs.end(), [](auto attr) {
    return attr.getName() == "generic";
  });
}

SmallVector<NamedAttribute> ComponentBuilder::Ctx::builtinAttrs(OpBuilder &builder) {
  return {NamedAttribute(builder.getStringAttr("builtin"), builder.getUnitAttr())};
}

SmallVector<NamedAttribute> ComponentBuilder::Ctx::funcBodyAttrs(OpBuilder &builder) {
  return {NamedAttribute(builder.getStringAttr("sym_visibility"), builder.getStringAttr("public"))};
}

void ComponentBuilder::Ctx::addBody(
    component::StructDefOp op, OpBuilder &builder, const TypeConverter &tc
) {
  body->set(op, *this, builder, tc);
}

void ComponentBuilder::Ctx::addFields(OpBuilder &builder) {
  for (auto &field : fields) {
    builder.create<component::FieldDefOp>(
        field.loc.value_or(builder.getUnknownLoc()), builder.getStringAttr(field.name),
        TypeAttr::get(field.type), bool(field.column)
    );
  }
}

void ComponentBuilder::Ctx::checkRequirements() { assert(body != nullptr); }

void ComponentBuilder::Ctx::checkBareRequirements() {
  assert(loc.has_value());
  assert(!compName.empty());
  assert((isBuiltin || isClosure) == compAttrs.empty());
  // If the component is a builtin it cannot use back-variables.
  if (isBuiltin) {
    assert(!usesBackVariables);
  }
}

ComponentBuilder::type ComponentBuilder::Ctx::buildBare(OpBuilder &builder) {
  auto builtin = builtinAttrs(builder);
  SmallVector<NamedAttribute> closure;
  ArrayRef<NamedAttribute> attrs;
  if (isClosure) {
    attrs = closure;
  } else {
    attrs = isBuiltin ? ArrayRef<NamedAttribute>(builtin) : compAttrs;
  }

  auto params = builder.getArrayAttr(llvm::map_to_vector(
      isGeneric() ? typeParams : ArrayRef<Identifier>(),
      [&builder](auto &s) -> Attribute { return builder.getStringAttr(s); }
  )

  );
  auto structOp = builder.create<component::StructDefOp>(*loc, compName, params);
  // Inject attributes zml requires for handling components.
  if (isClosure) {
    structOp->setDiscardableAttr("zml.closure", builder.getUnitAttr());
  } else if (isBuiltin) {
    structOp->setDiscardableAttr("zml.builtin", builder.getUnitAttr());
  } else {
    structOp->setDiscardableAttrs(compAttrs);
  }
  return structOp;
}

//===----------------------------------------------------------------------===//
// ComponentBuilder::BodySrc
//===----------------------------------------------------------------------===//

ComponentBuilder::BodySrc::~BodySrc() = default;

//===----------------------------------------------------------------------===//
// Body construction helpers
//===----------------------------------------------------------------------===//

static void deriveConstrainFnType(const FunctionType &src, FunctionType &dst, Builder &builder) {
  SmallVector<Type> args(src.getResults());
  args.insert(args.end(), src.getInputs().begin(), src.getInputs().end());
  dst = builder.getFunctionType(args, {});
}

static FunctionType convertFnType(FunctionType src, const TypeConverter &tc) {
  Builder builder(src.getContext());

  SmallVector<Type> inputs, results;
  assert(succeeded(tc.convertTypes(src.getInputs(), inputs)));
  assert(succeeded(tc.convertTypes(src.getResults(), results)));
  return builder.getFunctionType(inputs, results);
}

static void prepareFnTypes(
    FunctionType orig, FunctionType &compute, FunctionType &constrain, const TypeConverter &tc
) {
  Builder builder(orig.getContext());
  compute = convertFnType(orig, tc);
  deriveConstrainFnType(compute, constrain, builder);
}

static void fillFunc(
    OpBuilder &builder, const TypeConverter &tc, Location loc, StringRef name, FunctionType type,
    ArrayRef<NamedAttribute> attrs, ArrayRef<Location> argLocs, ComponentBuilder::FillFn fill
) {
  auto func = builder.create<function::FuncDefOp>(loc, name, type, attrs);
  OpBuilder::InsertionGuard insertionGuard(builder);
  auto *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  entryBlock->addArguments(type.getInputs(), argLocs);

  fill(func.getArguments(), builder, tc);
}

static void prepareArgLocs(
    OpBuilder &builder, size_t ctorArgsCount, std::optional<ArrayRef<Location>> argLocs,
    Location loc, SmallVectorImpl<Location> &ctorLocs, SmallVectorImpl<Location> &constrainLocs
) {
  if (argLocs.has_value()) {
    ctorLocs = SmallVector<Location>(*argLocs);
  } else {
    ctorLocs = SmallVector<Location>(ctorArgsCount, builder.getUnknownLoc());
  }
  constrainLocs = SmallVector<Location>({loc});
  constrainLocs.reserve(ctorLocs.size() + 1);
  constrainLocs.insert(constrainLocs.end(), ctorLocs.begin(), ctorLocs.end());
}

//===----------------------------------------------------------------------===//
// ComponentBuilder::FillBody
//===----------------------------------------------------------------------===//

ComponentBuilder::FillBody::FillBody(
    ArrayRef<Type> ArgTypes, ArrayRef<Type> Results, FillFn compute, FillFn constrain
)
    : argTypes(ArgTypes), results(Results), fillCompute(compute), fillConstrain(constrain) {}

void ComponentBuilder::FillBody::set(
    type op, Ctx &buildCtx, OpBuilder &builder, const TypeConverter &tc
) const {
  auto comp = cast<zml::ComponentInterface>(op.getOperation());

  prepareFnTypes(
      builder.getFunctionType(argTypes, results), buildCtx.constructorType,
      buildCtx.constrainFnType, tc
  );

  auto attrs = buildCtx.funcBodyAttrs(builder);
  auto loc = buildCtx.loc.value_or(op->getLoc());
  SmallVector<Location> ctorLocs, constrainLocs;
  prepareArgLocs(builder, argTypes.size(), buildCtx.argLocs, loc, ctorLocs, constrainLocs);

  fillFunc(
      builder, tc, loc, comp.getBodyFuncName(), buildCtx.constructorType, attrs, ctorLocs,
      fillCompute
  );

  fillFunc(
      builder, tc, loc, comp.getConstrainFuncName(), buildCtx.constrainFnType, attrs, constrainLocs,
      [&](auto args, auto &B, auto &TC) {
    if (fillConstrain) {
      fillConstrain(args, B, TC);
    } else {
      // Create an empty constrain function instead
      builder.create<function::ReturnOp>(loc);
    }
  }
  );
}

//===----------------------------------------------------------------------===//
// ComponentBuilder::TakeRegion
//===----------------------------------------------------------------------===//

ComponentBuilder::TakeRegion::TakeRegion(Region *bodyRegion) : body(bodyRegion) {}

void ComponentBuilder::TakeRegion::set(
    type op, Ctx &buildCtx, OpBuilder &builder, const TypeConverter &tc
) const {

  auto comp = cast<zml::ComponentInterface>(op.getOperation());
  assert(buildCtx.constructorType);
  assert(body);
  prepareFnTypes(buildCtx.constructorType, buildCtx.constructorType, buildCtx.constrainFnType, tc);

  Region constrainBody;
  IRMapping mapper;
  // Create a copy of the body to give to the constrain function.
  // Necessary since the we move the region.
  body->cloneInto(&constrainBody, mapper);

  auto loc = buildCtx.loc.value_or(op.getLoc());
  auto zmlType = comp.getType();
  auto llzkType = tc.convertType(zmlType);
  auto attrs = buildCtx.funcBodyAttrs(builder);

  auto ctorArgsCount = buildCtx.constructorType.getNumInputs();
  SmallVector<Location> ctorLocs, constrainLocs;
  prepareArgLocs(builder, ctorArgsCount, buildCtx.argLocs, loc, ctorLocs, constrainLocs);

  auto filler = [location = loc, zmlType,
                 llzkType](Region *region, ValueRange, OpBuilder &B, const TypeConverter &) {
    auto zmlSelf = B.create<zml::SelfOp>(location, zmlType, *region);
    auto llzkSelf = B.create<UnrealizedConversionCastOp>(location, llzkType, zmlSelf.getResult());
    B.create<function::ReturnOp>(location, ValueRange({llzkSelf.getResult(0)}));
  };

  fillFunc(
      builder, tc, loc, comp.getBodyFuncName(), buildCtx.constructorType, attrs, ctorLocs,
      std::bind_front(filler, body)
  );

  fillFunc(
      builder, tc, loc, comp.getConstrainFuncName(), buildCtx.constrainFnType, attrs, constrainLocs,
      std::bind_front(filler, &constrainBody)
  );
}

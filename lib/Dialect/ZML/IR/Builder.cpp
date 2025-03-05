#include <functional>
#include <zklang/Dialect/ZML/IR/Builder.h>

namespace zml {

ComponentBuilder::BodySrc::~BodySrc() = default;

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

std::vector<mlir::NamedAttribute> ComponentBuilder::Ctx::builtinAttrs(mlir::OpBuilder &builder) {
  return {mlir::NamedAttribute(builder.getStringAttr("builtin"), builder.getUnitAttr())};
}
std::vector<mlir::NamedAttribute> ComponentBuilder::Ctx::funcBodyAttrs(mlir::OpBuilder &builder) {
  return {
      mlir::NamedAttribute(builder.getStringAttr("sym_visibility"), builder.getStringAttr("public"))
  };
}
void ComponentBuilder::Ctx::addBody(ComponentOp op, mlir::OpBuilder &builder) {
  body->set(op, *this, builder);
}
void ComponentBuilder::Ctx::addFields(mlir::OpBuilder &builder) {
  for (auto &field : fields) {
    builder.create<FieldDefOp>(
        field.loc.value_or(builder.getUnknownLoc()), builder.getStringAttr(field.name),
        mlir::TypeAttr::get(field.type)
    );
  }
}
void ComponentBuilder::Ctx::checkRequirements() { assert(body != nullptr); }

void ComponentBuilder::Ctx::checkBareRequirements() {
  assert(loc.has_value());
  assert(!compName.empty());
  assert((isBuiltin || isClosure) == compAttrs.empty());
}

ComponentOp ComponentBuilder::Ctx::buildBare(mlir::OpBuilder &builder) {
  auto builtin = builtinAttrs(builder);
  std::vector<mlir::NamedAttribute> closure;
  mlir::ArrayRef<mlir::NamedAttribute> attrs;
  if (isClosure) {
    attrs = closure;
  } else {
    attrs = isBuiltin ? mlir::ArrayRef<mlir::NamedAttribute>(builtin) : compAttrs;
  }

  if (isGeneric()) {
    return builder.create<ComponentOp>(*loc, compName, typeParams, attrs);
  } else {
    return builder.create<ComponentOp>(*loc, compName, attrs);
  }
}

ComponentBuilder::TakeRegion::TakeRegion(mlir::Region *body) : body(body) {}

ComponentOp ComponentBuilder::build(mlir::OpBuilder &builder) {
  if (!ctx.loc.has_value()) {
    ctx.loc = builder.getUnknownLoc();
  }
  ctx.checkBareRequirements();
  auto op = ctx.buildBare(builder);
  if (ctx.deferCb) {
    ctx.deferCb(op);
  }

  ctx.checkRequirements();

  mlir::OpBuilder::InsertionGuard guard(builder);
  auto block = builder.createBlock(&op.getRegion());
  builder.setInsertionPointToStart(block);
  ctx.addFields(builder);
  ctx.addBody(op, builder);

  ctx = Ctx();
  return op;
}

ComponentBuilder &ComponentBuilder::typeParams(mlir::ArrayRef<std::string> params) {
  for (auto &name : params) {
    typeParam(name);
  }
  return *this;
}
ComponentBuilder &ComponentBuilder::typeParam(mlir::StringRef param) {
  ctx.typeParams.push_back(param);
  return *this;
}
ComponentBuilder &ComponentBuilder::attrs(mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  ctx.compAttrs = attrs;
  return *this;
}
ComponentBuilder &ComponentBuilder::field(std::string_view name, mlir::Type type) {
  ctx.fields.push_back({.name = name, .type = type, .loc = std::nullopt});
  return *this;
}
ComponentBuilder &
ComponentBuilder::field(std::string_view name, mlir::Type type, mlir::Location loc) {
  ctx.fields.push_back({.name = name, .type = type, .loc = loc});
  return *this;
}
ComponentBuilder &ComponentBuilder::name(std::string_view name) {
  ctx.compName = name;
  return *this;
}
ComponentBuilder &ComponentBuilder::location(mlir::Location loc) {
  ctx.loc = loc;
  return *this;
}
ComponentBuilder &ComponentBuilder::constructor(
    mlir::FunctionType constructorType, std::vector<mlir::Location> argLocs
) {
  ctx.constructorType = constructorType;
  ctx.argLocs = argLocs;
  return *this;
}
ComponentBuilder &ComponentBuilder::fillBody(
    mlir::ArrayRef<mlir::Type> argTypes, mlir::ArrayRef<mlir::Type> results,
    std::function<void(mlir::ValueRange, mlir::OpBuilder &)> fn
) {
  ctx.body = std::make_unique<FillBody>(argTypes, results, fn);
  return *this;
}
ComponentBuilder &ComponentBuilder::takeRegion(mlir::Region *region) {
  ctx.body = std::make_unique<TakeRegion>(region);
  return *this;
}
ComponentBuilder &ComponentBuilder::isBuiltin() {
  ctx.isBuiltin = true;
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

ComponentBuilder &ComponentBuilder::defer(std::function<void(ComponentOp)> cb) {
  ctx.deferCb = cb;
  return *this;
}

void ComponentBuilder::FillBody::set(ComponentOp op, Ctx &buildCtx, mlir::OpBuilder &builder)
    const {
  buildCtx.constructorType = builder.getFunctionType(argTypes, results);
  std::vector<mlir::NamedAttribute> attrs = buildCtx.funcBodyAttrs(builder);

  auto bodyOp = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), op.getBodyFuncName(), buildCtx.constructorType, attrs
  );
  mlir::OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToStart(bodyOp.addEntryBlock());
  delegate(bodyOp.getArguments(), builder);
}

ComponentBuilder::FillBody::FillBody(
    mlir::ArrayRef<mlir::Type> argTypes, mlir::ArrayRef<mlir::Type> results,
    std::function<void(mlir::ValueRange, mlir::OpBuilder &)> delegate
)
    : argTypes(argTypes), results(results), delegate(delegate) {}

void ComponentBuilder::TakeRegion::set(ComponentOp op, Ctx &buildCtx, mlir::OpBuilder &builder)
    const {
  auto constructorAttrs = buildCtx.funcBodyAttrs(builder);

  assert(buildCtx.constructorType);
  assert(body);
  auto bodyOp = builder.create<mlir::func::FuncOp>(
      *buildCtx.loc, op.getBodyFuncName(), buildCtx.constructorType, constructorAttrs
  );

  // Create arguments for the entry block (aka region arguments)
  auto &entryBlock = bodyOp.getRegion().emplaceBlock();
  assert(bodyOp.getRegion().hasOneBlock());
  entryBlock.addArguments(buildCtx.constructorType.getInputs(), buildCtx.argLocs);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entryBlock);

  auto self = builder.create<zml::SelfOp>(op.getLoc(), op.getType(), *body);
  builder.create<mlir::func::ReturnOp>(self.getLoc(), mlir::ValueRange({self}));
}

} // namespace zml

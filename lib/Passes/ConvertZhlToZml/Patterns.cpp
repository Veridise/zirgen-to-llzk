#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Index/IR/IndexAttrs.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <vector>
#include <zklang/Dialect/ZHL/Typing/ArrayFrame.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameImpl.h>
#include <zklang/Dialect/ZHL/Typing/InnerFrame.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>
#include <zklang/Dialect/ZML/IR/Builder.h>
#include <zklang/Dialect/ZML/IR/OpInterfaces.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/Types.h>
#include <zklang/Dialect/ZML/Typing/Materialize.h>
#include <zklang/Dialect/ZML/Typing/ZMLTypeConverter.h>
#include <zklang/Passes/ConvertZhlToZml/Helpers.h>
#include <zklang/Passes/ConvertZhlToZml/Patterns.h>

using namespace zirgen;
using namespace zhl;
using namespace mlir;
using namespace zml;

///////////////////////////////////////////////////////////
/// ZhlLiteralLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlLiteralLowering::matchAndRewrite(
    Zhl::LiteralOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }

  auto val = materializeTypeBinding(getContext(), *binding);
  rewriter.replaceOpWithNewOp<LitValOp>(op, val, adaptor.getValue());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlLiteralStrLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlLiteralStrLowering::matchAndRewrite(
    Zhl::StringOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }
  auto val = materializeTypeBinding(getContext(), *binding);
  rewriter.replaceOpWithNewOp<LitStrOp>(op, val, adaptor.getValue());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlParameterLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlParameterLowering::matchAndRewrite(
    Zhl::ConstructorParamOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto body = op->getParentOfType<mlir::func::FuncOp>();
  mlir::BlockArgument arg = body.getArgument(adaptor.getIndex());

  rewriter.replaceOpWithNewOp<NopOp>(op, arg.getType(), arg);
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlConstructLowering
///////////////////////////////////////////////////////////

inline bool validArgCount(bool isVariadic, size_t argCount, size_t formalsCount) {
  if (isVariadic) {
    // For variadics it's only valid to have from formalsCount-1 arguments and onwards.
    return argCount >= formalsCount - 1;
  }
  // For non variadics the number of arguments must be equal to the number of formals
  return argCount == formalsCount;
}

inline bool isVariadic(mlir::ArrayRef<mlir::Type> ctorFormals) {
  return !ctorFormals.empty() && mlir::isa<VarArgsType>(ctorFormals.back());
}

inline bool isVariadic(mlir::FunctionType fnType) { return isVariadic(fnType.getInputs()); }

mlir::LogicalResult ZhlConstructLowering::matchAndRewrite(
    Zhl::ConstructOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto ctor = makeCtorCallBuilder(op, op, rewriter);
  if (failed(ctor)) {
    return failure(); // makeCtorCallBuilder emits an error message already
  }
  auto &binding = ctor->getBinding();
  auto constructorType = ctor->getCtorType();
  auto ctorFormalsCount = constructorType.getInputs().size();

  if (!validArgCount(isVariadic(constructorType), adaptor.getArgs().size(), ctorFormalsCount)) {
    // Depending if it's variadic or not the message changes a bit.
    StringRef expectingNArgsMsg =
        isVariadic(constructorType) ? ", was expecting at least " : ", was expecting ";
    auto minArgCount = isVariadic(constructorType) ? ctorFormalsCount - 1 : ctorFormalsCount;
    return op->emitOpError().append(
        "incorrect number of arguments for component ", binding.getName(), expectingNArgsMsg,
        minArgCount, " arguments but got ", adaptor.getArgs().size()
    );
  }

  std::vector<mlir::Value> preparedArguments;
  prepareArguments(
      adaptor.getArgs(), constructorType.getInputs(), op->getLoc(), rewriter, preparedArguments
  );

  auto result = ctor->build(rewriter, op.getLoc(), preparedArguments);
  rewriter.replaceOp(op, result);
  return mlir::success();
}

mlir::ValueRange::iterator
computeVarArgsSplice(mlir::ValueRange &args, mlir::ArrayRef<mlir::Type> constructorTypes) {
  if (!isVariadic(constructorTypes)) {
    return args.end();
  }

  // If the call does not have any varargs then there are less elements in `args` than there is in
  // `constructorTypes`. This comparison is safe because the call has already been type checked by
  // this point.
  if (args.size() < constructorTypes.size()) {
    return args.end();
  }

  // Compute how many arguments are var-args and substract that amount to the number of arguments to
  // get the offset from args.begin() where the var-args start.
  auto C = constructorTypes.size();
  auto offset = C - 1;
  return args.begin() + offset;
}

void ZhlConstructLowering::prepareArguments(
    mlir::ValueRange args, mlir::ArrayRef<mlir::Type> constructorTypes, mlir::Location loc,
    mlir::ConversionPatternRewriter &rewriter, std::vector<mlir::Value> &preparedArgs
) const {

  preparedArgs.clear();
  preparedArgs.reserve(constructorTypes.size());

  // Compute the point where the normal arguments end and the var-args begin
  auto argsEnd = computeVarArgsSplice(args, constructorTypes);
  // Prepare the arguments up to the splice point. These arguments are passed as is since they are
  // normal arguments.
  std::transform(
      args.begin(), argsEnd, constructorTypes.begin(), std::back_inserter(preparedArgs),
      [&](mlir::Value v, mlir::Type t) { return prepareArgument(v, t, loc, rewriter); }
  );
  if (isVariadic(constructorTypes)) {
    std::vector<mlir::Value> vargs;
    auto varType = mlir::dyn_cast<VarArgsType>(constructorTypes.back());
    assert(varType && "expecting a var args type");

    // If there aren't any arguments left to prepare then argsEnd == args.end() and this call is a
    // no-op.
    std::transform(argsEnd, args.end(), std::back_inserter(vargs), [&](mlir::Value v) {
      return prepareArgument(v, varType.getInner(), loc, rewriter);
    });

    // Regardless of what std::transform does above add the VarArgsOp to make sure we have the
    // correct number of arguments.
    auto va = rewriter.create<VarArgsOp>(loc, constructorTypes.back(), vargs);
    preparedArgs.push_back(va);
  }

  assert(preparedArgs.size() == constructorTypes.size() && "incorrect number of arguments");
}

mlir::Value ZhlConstructLowering::prepareArgument(
    mlir::Value arg, mlir::Type expectedType, mlir::Location loc,
    mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(arg);
  auto type = expectedType;

  if (mlir::succeeded(binding)) {
    type = materializeTypeBinding(rewriter.getContext(), *binding);
  }
  if (arg.getType() == type) {
    return arg;
  }
  auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, type, arg);
  if (expectedType == cast.getResult(0).getType()) {
    return cast.getResult(0);
  }
  return rewriter.create<SuperCoerceOp>(loc, expectedType, cast.getResult(0));
}

mlir::LogicalResult ZhlConstrainLowering::matchAndRewrite(
    Zhl::ConstraintOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (failed(binding)) {
    return failure();
  }
  auto lhsBinding = getType(op.getLhs());
  if (mlir::failed(lhsBinding)) {
    return op->emitOpError() << "failed to type check lhs";
  }
  auto rhsBinding = getType(op.getRhs());
  if (mlir::failed(rhsBinding)) {
    return op->emitOpError() << "failed to type check rhs";
  }

  auto constraintType = materializeTypeBinding(getContext(), *binding);
  mlir::Value lhsValue = getCastedValue(adaptor.getLhs(), *lhsBinding, rewriter, constraintType);

  mlir::Value rhsValue = getCastedValue(adaptor.getRhs(), *rhsBinding, rewriter, constraintType);

  rewriter.replaceOpWithNewOp<ConstrainOp>(op, lhsValue, rhsValue);
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlGlobalRemoval
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlGlobalRemoval::matchAndRewrite(
    zirgen::Zhl::GlobalOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }
  rewriter.replaceOpWithNewOp<NopOp>(
      op, mlir::TypeRange({materializeTypeBinding(getContext(), *binding)}), mlir::ValueRange()
  );
  return mlir::success();
}

LogicalResult ZhlDirectiveRemoval::matchAndRewrite(
    zirgen::Zhl::DirectiveOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.eraseOp(op);
  return success();
}

///////////////////////////////////////////////////////////
/// ZhlGenericRemoval
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlGenericRemoval::matchAndRewrite(
    zirgen::Zhl::TypeParamOp op, OpAdaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }
  auto type = materializeTypeBinding(getContext(), *binding);
  if (binding->isGenericParam() && binding->getSuperType().isVal()) {
    rewriter.replaceOpWithNewOp<LoadValParamOp>(
        op, ComponentType::Val(getContext()), op.getNameAttr()
    );
  } else {
    rewriter.replaceOpWithNewOp<NopOp>(op, mlir::TypeRange({type}), mlir::ValueRange());
  }
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlSpecializeRemoval
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlSpecializeRemoval::matchAndRewrite(
    zirgen::Zhl::SpecializeOp op, OpAdaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }
  rewriter.replaceOpWithNewOp<NopOp>(
      op, mlir::TypeRange({materializeTypeBinding(getContext(), *binding)}), mlir::ValueRange()
  );
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlDeclarationRemoval
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlDeclarationRemoval::matchAndRewrite(
    zirgen::Zhl::DeclarationOp op, OpAdaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<NopOp>(
      op, mlir::TypeRange({ComponentType::Component(rewriter.getContext())}), mlir::ValueRange()
  );
  return mlir::success();
}

inline bool opConstructsComponent(mlir::Operation *op) {
  // TODO: Add the rest of the ops: array constructors, for loops, etc
  return mlir::isa<func::CallIndirectOp>(op);
}

///////////////////////////////////////////////////////////
/// ZhlDefinitionLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlDefineLowering::matchAndRewrite(
    zirgen::Zhl::DefinitionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitError() << "failed to type check";
  }
  auto declrBinding = getType(op.getDeclaration());
  if (failed(declrBinding)) {
    return op->emitError() << "failed to type check declaration expression";
  }
  auto exprBinding = getType(op.getDefinition());
  if (failed(exprBinding)) {
    return op->emitError() << "failed to type check definition expression";
  }
  ComponentSlot *slot = nullptr;
  if (binding->getSlot()) {
    if (auto *thisSlot = dyn_cast<ComponentSlot>(binding->getSlot())) {
      slot = thisSlot;
    }
  } else if (declrBinding->getSlot()) {
    if (auto *declrSlot = dyn_cast<ComponentSlot>(declrBinding->getSlot())) {
      slot = declrSlot;
    }
  }

  auto comp = op->getParentOfType<ComponentInterface>();
  assert(comp);
  auto self = op->getParentOfType<SelfOp>().getSelfValue();

  mlir::Value value = adaptor.getDefinition();

  // If the binding of this op has a slot then it is responsible of creating it.
  // Otherwise, check if the declaration's binding has a slot. If it does create it here.
  // Wrap the value around a SuperCoerceOp.
  // If a slot is created here that means that the value we are storing in the field does not need
  // memory according to the type checker and thus we don't need to introduce the use-def cut since
  // that value is safe to use in @constrain functions.

  if (slot) {
    auto slotName = createSlot(slot, rewriter, comp, op.getLoc());
    mlir::Type slotType = materializeTypeBinding(getContext(), slot->getBinding());
    SmallVector<Operation *, 2> castOps;
    Value result = getCastedValue(value, *exprBinding, rewriter, castOps);
    storeSlot(*slot, result, slotName, slotType, op.getLoc(), comp.getType(), rewriter, self);
  }

  rewriter.eraseOp(op);
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlSuperLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlSuperLoweringInFunc::matchAndRewrite(
    zirgen::Zhl::SuperOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }
  if (!mlir::isa<SelfOp>(op->getParentOp())) {
    return failure();
  }
  auto comp = op->getParentOfType<ComponentInterface>();
  assert(comp);
  auto self = op->getParentOfType<SelfOp>().getSelfValue();
  mlir::Type target = materializeTypeBinding(getContext(), *binding);
  auto value = getCastedValue(adaptor.getValue(), rewriter, target);
  if (mlir::failed(value)) {
    return mlir::failure();
  }

  rewriter.replaceOpWithNewOp<WriteFieldOp>(op, self, "$super", *value);
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlExternLowering
///////////////////////////////////////////////////////////

std::vector<mlir::NamedAttribute> externFuncAttrs(mlir::ConversionPatternRewriter &rewriter) {
  return {
      mlir::NamedAttribute(rewriter.getStringAttr("extern"), rewriter.getUnitAttr()),
      mlir::NamedAttribute(
          rewriter.getStringAttr("sym_visibility"), rewriter.getStringAttr("private")
      )
  };
}

mlir::FailureOr<mlir::func::FuncOp> createExternFunc(
    ComponentInterface op, Twine &name, mlir::FunctionType type,
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc
) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);
  auto attrs = externFuncAttrs(rewriter);
  SmallVector<char, 10> nameMem;
  return rewriter.create<mlir::func::FuncOp>(loc, name.toStringRef(nameMem), type, attrs);
}

mlir::LogicalResult ZhlExternLowering::matchAndRewrite(
    Zhl::ExternOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }
  auto comp = op->getParentOfType<ComponentOp>();
  if (!comp) {
    return mlir::failure();
  }

  std::vector<mlir::Type> argBindings;
  for (auto arg : op.getArgs()) {
    auto argBinding = getType(arg);
    if (mlir::failed(argBinding)) {
      return op->emitOpError() << "failed to type check argument #" << argBindings.size();
    }
    argBindings.push_back(materializeTypeBinding(getContext(), *argBinding));
  }

  auto retType = materializeTypeBinding(getContext(), *binding);

  // Extern ops are wrapped around a component by the AST->ZHL step and have the same inputs as the
  // component.
  auto funcType =
      rewriter.getFunctionType(comp.getBodyFunc().getFunctionType().getInputs(), {retType});
  Twine externName = op.getName() + "$$extern";
  auto externDeclrResult = createExternFunc(comp, externName, funcType, rewriter, op.getLoc());
  if (failed(externDeclrResult)) {
    return failure();
  }
  auto externNameSymRef = SymbolRefAttr::get(rewriter.getStringAttr(externName));

  auto externFnRef = rewriter.create<ExternFnRefOp>(op.getLoc(), funcType, externNameSymRef);
  std::vector<mlir::Value> args;
  std::transform(
      adaptor.getArgs().begin(), adaptor.getArgs().end(), argBindings.begin(),
      std::back_inserter(args),
      [&](mlir::Value adaptorValue, mlir::Type argType) {
    if (adaptorValue.getType() == argType) {
      return adaptorValue;
    }

    auto cast =
        rewriter.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), argType, adaptorValue);
    return cast.getResult(0);
  }
  );

  rewriter.replaceOpWithNewOp<mlir::func::CallIndirectOp>(op, externFnRef, mlir::ValueRange(args));

  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlLookupLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlLookupLowering::matchAndRewrite(
    Zhl::LookupOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto comp = adaptor.getComponent();
  auto originalComp = getType(op.getComponent());
  if (mlir::failed(originalComp)) {
    return op->emitOpError() << "failed to type check component reference";
  }
  auto materializedType = materializeTypeBinding(getContext(), *originalComp);
  auto compType = mlir::dyn_cast<ComponentType>(materializedType);
  if (!compType) {
    return op->emitError() << "type mismatch, cannot access a member for a non-component type "
                           << materializedType;
  }
  if (comp.getType() != compType) {
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), compType, comp);
    comp = cast.getResult(0);
  }
  mlir::SymbolTableCollection st;
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  auto compDef = compType.getDefinition(st, mod);
  assert(compDef && "Component type without a definition!");

  auto nameSym = mlir::SymbolRefAttr::get(adaptor.getMemberAttr());
  while (mlir::failed(compDef.lookupFieldType(nameSym))) {
    auto superType = compType.getSuperType();
    if (!superType) {
      return op->emitError() << "member " << adaptor.getMember() << " was not found";
    }
    compType = mlir::dyn_cast<ComponentType>(superType);
    if (!compType) {
      return op->emitError() << "type mismatch, cannot access a member for a non-component type "
                             << superType;
    }

    compDef = compType.getDefinition(st, mod);
  }

  auto fieldType = compDef.lookupFieldType(nameSym);
  assert(mlir::succeeded(fieldType));

  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }
  auto bindingType = materializeTypeBinding(getContext(), *binding);
  if (*fieldType != bindingType) {
    return op->emitError() << "type mismatch, was expecting " << bindingType << " but field "
                           << adaptor.getMember() << " is of type " << *fieldType;
  }

  // Coerce to the type in the chain that defines the accessed member
  comp = rewriter.create<SuperCoerceOp>(op.getLoc(), compType, comp);

  rewriter.replaceOpWithNewOp<ReadFieldOp>(op, bindingType, comp, adaptor.getMember());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlSubscriptLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlSubscriptLowering::matchAndRewrite(
    Zhl::SubscriptOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }

  auto arrayBinding = getType(op.getArray());
  if (mlir::failed(arrayBinding)) {
    return op->emitOpError() << "failed to type check array reference";
  }
  auto elementBinding = getType(op.getElement());
  if (mlir::failed(elementBinding)) {
    return op->emitOpError() << "failed to type check index";
  }
  auto concreteArrayTypeBinding = arrayBinding->getConcreteArrayType();
  if (failed(concreteArrayTypeBinding)) {
    return failure();
  }

  Type concreteArrayType = zml::materializeTypeBinding(getContext(), *concreteArrayTypeBinding);
  auto arrayVal = getCastedValue(adaptor.getArray(), *arrayBinding, rewriter, concreteArrayType);
  auto Val = ComponentType::Val(getContext());
  auto elementVal = getCastedValue(adaptor.getElement(), *elementBinding, rewriter, Val);

  rewriter.replaceOpWithNewOp<zml::ReadArrayOp>(
      op, zml::materializeTypeBinding(getContext(), *binding), arrayVal, elementVal

  );
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlArrayLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlArrayLowering::matchAndRewrite(
    Zhl::ArrayOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }

  if (!binding->isArray()) {
    return op->emitOpError() << "was expecting an 'Array' component but got '" << binding->getName()
                             << "'";
  }
  auto arrayType = materializeTypeBinding(getContext(), *binding);
  auto elementTypeBinding = binding->getArrayElement([&]() { return op->emitOpError(); });
  if (mlir::failed(elementTypeBinding)) {
    return mlir::failure();
  }
  auto elementType = materializeTypeBinding(getContext(), *elementTypeBinding);

  llvm::SmallVector<FailureOr<TypeBinding>> argBindings;
  std::transform(
      op.getElements().begin(), op.getElements().end(), std::back_inserter(argBindings),
      [&](auto element) { return getType(element); }
  );

  if (std::any_of(argBindings.begin(), argBindings.end(), failed)) {
    return op->emitOpError() << "failed to type check array values";
  }

  if (adaptor.getElements().empty()) {
    assert(false && "TODO");

    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> args;

  std::transform(
      adaptor.getElements().begin(), adaptor.getElements().end(), argBindings.begin(),
      std::back_inserter(args),
      [&](auto element, auto eltBinding) -> Value {
    return getCastedValue(element, *eltBinding, rewriter, elementType);
  }
  );
  rewriter.replaceOpWithNewOp<NewArrayOp>(op, arrayType, args);
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlCompToZmirCompPattern
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlCompToZmirCompPattern::matchAndRewrite(
    zirgen::Zhl::ComponentOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto name = getType(op.getName());
  if (mlir::failed(name)) {
    return op->emitOpError() << "could not be lowered because its type could not be infered";
  }

  ComponentBuilder builder;
  auto genericNames = name->getGenericParamNames();
  builder.name(name->getName())
      .location(op->getLoc())
      .attrs(op->getAttrs())
      .typeParams(genericNames)
      .constructor(
          materializeTypeBindingConstructor(rewriter, *name), name->getConstructorParamLocations()
      )
      .takeRegion(&op.getRegion());
  for (auto &[fieldName, binding] : name->getMembers()) {
    if (!binding.has_value()) {
      return op->emitOpError() << "failed to type check component member '" << fieldName << "'";
    }
  }
  auto &super = name->getSuperType();
  builder.field("$super", materializeTypeBinding(getContext(), super));
  auto maybeBuiltin = mlir::SymbolTable::lookupSymbolIn(
      op->getParentOfType<mlir::ModuleOp>().getOperation(), op.getName()
  );

  if (maybeBuiltin) {
    rewriter.eraseOp(maybeBuiltin);
    builtinOverriden(op.getName());
  }
  auto comp = builder.build(rewriter);

  rewriter.replaceOp(op.getOperation(), comp.getOperation());

  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlRangeOpLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlRangeOpLowering::matchAndRewrite(
    Zhl::RangeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }
  assert(binding->isArray());
  auto startBinding = getType(op.getStart());
  if (mlir::failed(startBinding)) {
    return op->emitOpError() << "failed to type check start";
  }
  auto endBinding = getType(op.getEnd());
  if (mlir::failed(endBinding)) {
    return op->emitOpError() << "failed to type check end";
  }
  auto type = materializeTypeBinding(getContext(), *binding);
  auto innerBinding = binding->getArrayElement([&]() { return op->emitError(); });
  if (mlir::failed(innerBinding)) {
    return mlir::failure();
  }

  // Create a literal array directly if we know the range is made of literal values
  if (startBinding->isKnownConst() && endBinding->isKnownConst()) {
    SmallVector<int64_t> values;
    unsigned E = endBinding->getConst();
    unsigned I = startBinding->getConst();
    values.reserve(E - I);
    for (; I < E; I++) {
      values.push_back(I);
    }
    auto litArr = DenseI64ArrayAttr::get(getContext(), values);
    rewriter.replaceOpWithNewOp<LitValArrayOp>(
        op, ComponentType::Array(getContext(), ComponentType::Val(getContext()), values.size()),
        litArr
    );
    return success();
  }
  auto innerType = materializeTypeBinding(getContext(), *innerBinding);
  Value startVal =
      getCastedValue(adaptor.getStart(), *startBinding, rewriter, ComponentType::Val(getContext()));

  Value endVal =
      getCastedValue(adaptor.getEnd(), *endBinding, rewriter, ComponentType::Val(getContext()));

  auto arrAlloc = rewriter.create<AllocArrayOp>(op.getLoc(), type);

  // Create a for loop op using the operands as bounds
  auto one = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), 1);
  auto start = rewriter.create<ValToIndexOp>(op.getStart().getLoc(), startVal);
  auto end = rewriter.create<ValToIndexOp>(op.getEnd().getLoc(), endVal);
  rewriter.create<mlir::scf::ForOp>(
      op.getLoc(), start, end, one, mlir::ValueRange(),
      [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange args) {
    auto conv = builder.create<IndexToValOp>(loc, innerType, iv);
    builder.create<WriteArrayOp>(loc, arrAlloc, iv, conv);
    builder.create<scf::YieldOp>(loc);
  }
  );
  rewriter.replaceOp(op, arrAlloc);

  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlMapOpLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlMapLowering::matchAndRewrite(
    Zhl::MapOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }
  if (!binding->isArray()) {
    return op->emitOpError() << "was expecting 'Array' but got '" << binding->getName() << "'";
  }
  assert(binding->getSlot());
  auto *arrayFrame = cast<ArrayFrame>(binding->getSlot());

  auto inputBinding = getType(op.getArray());
  if (mlir::failed(inputBinding)) {
    return op->emitOpError() << "failed to type check input";
  }
  if (!inputBinding->isArray()) {
    return op->emitOpError() << "was expecting 'Array' but got '" << inputBinding->getName() << "'";
  }
  auto innerInputBinding = inputBinding->getArrayElement([&]() { return op->emitError(); });
  if (mlir::failed(innerInputBinding)) {
    return mlir::failure();
  }

  auto itType = materializeTypeBinding(getContext(), *innerInputBinding);
  auto outputType = materializeTypeBinding(getContext(), *binding);

  auto arrValue = getCastedValue(adaptor.getArray(), rewriter);
  assert(succeeded(arrValue) && "this binding was validated above");

  auto arrAlloc = rewriter.create<AllocArrayOp>(op.getLoc(), outputType);
  auto one = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), 1);
  auto zero = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), 0);
  auto len = rewriter.create<GetArrayLenOp>(op.getLoc(), *arrValue);

  auto loop = rewriter.create<mlir::scf::ForOp>(
      op.getLoc(), zero, len->getResult(0), one, mlir::ValueRange(arrAlloc),
      [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange args) {
    arrayFrame->setInductionVar(iv);
    auto itVal = builder.create<ReadArrayOp>(loc, itType, *arrValue, mlir::ValueRange(iv));
    // Cast it to a zhl Expr type for the block inlining
    auto itValCast = builder.create<mlir::UnrealizedConversionCastOp>(
        loc, mlir::TypeRange(Zhl::ExprType::get(getContext())), mlir::ValueRange(itVal)
    );
    auto loopPrologue = builder.getInsertionBlock();

    rewriter.inlineBlockBefore(
        &op.getRegion().front(), loopPrologue, loopPrologue->end(),
        mlir::ValueRange(itValCast.getResult(0))
    );
  }
  );

  rewriter.replaceOp(op, arrAlloc);
  loop->setAttr("original_op", rewriter.getStringAttr("map"));

  return mlir::success();
}

mlir::LogicalResult ZhlBlockLowering::matchAndRewrite(
    Zhl::BlockOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }

  auto type = materializeTypeBinding(getContext(), *binding);
  auto exec = rewriter.replaceOpWithNewOp<mlir::scf::ExecuteRegionOp>(op, type);
  rewriter.inlineRegionBefore(op.getRegion(), exec.getRegion(), exec.getRegion().end());
  return mlir::success();
}

mlir::LogicalResult ZhlSuperLoweringInMap::matchAndRewrite(
    Zhl::SuperOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto parent = op->getParentOp();
  if (!parent || !mlir::isa<mlir::scf::ForOp>(parent)) {
    return mlir::failure();
  }

  auto loopOp = mlir::cast<mlir::scf::ForOp>(parent);
  auto loopOriginalOp = loopOp->getAttr("original_op");
  if (!loopOriginalOp || loopOriginalOp != rewriter.getStringAttr("map")) {
    return mlir::failure();
  }
  auto value = getCastedValue(adaptor.getValue(), rewriter);
  if (failed(value)) {
    return op->emitError() << "failed to type check super value";
  }

  auto iv = loopOp.getInductionVar();
  auto arr = loopOp.getRegionIterArgs().front();

  rewriter.create<WriteArrayOp>(op.getLoc(), arr, iv, *value);
  rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, loopOp.getRegionIterArgs());

  return mlir::success();
}

mlir::LogicalResult ZhlSuperLoweringInBlock::matchAndRewrite(
    Zhl::SuperOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto parent = op->getParentOp();
  if (!parent || !mlir::isa<mlir::scf::ExecuteRegionOp>(parent)) {
    return failure();
  }
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }

  auto value = adaptor.getValue();
  auto type = materializeTypeBinding(getContext(), *binding);
  if (value.getType() != type) {
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), type, value);
    value = cast.getResult(0);
  }

  rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, value);
  return mlir::success();
}

mlir::LogicalResult ZhlSuperLoweringInSwitch::matchAndRewrite(
    Zhl::SuperOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto parent = op->getParentOp();
  if (!parent || !mlir::isa<mlir::scf::IfOp>(parent)) {
    return mlir::failure();
  }
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }

  auto type = materializeTypeBinding(getContext(), *binding);
  assert(parent->getResultTypes().size() == 1);
  auto parentType = parent->getResultTypes().front();
  auto value = getCastedValue(adaptor.getValue(), rewriter, type);
  if (failed(value)) {
    return failure();
  }
  auto coercion = rewriter.create<SuperCoerceOp>(op.getLoc(), parentType, *value);

  rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, coercion.getResult());
  return mlir::success();
}

mlir::LogicalResult ZhlReduceLowering::matchAndRewrite(
    Zhl::ReduceOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }

  assert(binding->getSlot());
  auto *arrayFrame = cast<ArrayFrame>(binding->getSlot());
  auto inputBinding = getType(op.getArray());
  if (mlir::failed(inputBinding)) {
    return op->emitOpError() << "failed to type check input";
  }
  if (!inputBinding->isArray()) {
    return op->emitOpError() << "was expecting 'Array' but got '" << inputBinding->getName() << "'";
  }
  auto innerInputBinding = inputBinding->getArrayElement([&]() { return op->emitError(); });
  if (mlir::failed(innerInputBinding)) {
    return mlir::failure();
  }
  auto accBinding = getType(op.getType());
  if (mlir::failed(accBinding)) {
    return op->emitOpError() << "failed to type check accumulator";
  }
  // Allocate a component slot for the output of the accumulator.
  auto self = op->getParentOfType<SelfOp>().getSelfValue();
  arrayFrame->getFrame().allocateSlot<ComponentSlot>(getTypeBindings(), *accBinding);

  auto ctorBuilder = CtorCallBuilder::Make(op, *accBinding, rewriter, self);
  if (mlir::failed(ctorBuilder)) {
    return mlir::failure();
  }
  auto rootModule = op->getParentOfType<mlir::ModuleOp>();
  auto *calleeComp = findCallee(accBinding->getName(), rootModule);
  if (!calleeComp) {
    return op->emitError() << "could not find component with name " << accBinding->getName();
  }

  auto constructorType = materializeTypeBindingConstructor(rewriter, *accBinding);
  assert(constructorType);
  if (constructorType.getInputs().size() != 2) {
    return op->emitOpError() << "was expecting a constructor with two arguments but got "
                             << constructorType.getInputs().size() << " arguments";
  }

  auto init = getCastedValue(adaptor.getInit(), rewriter);
  if (failed(init)) {
    return op->emitError() << "failed to type cast init value";
  }
  auto itType = materializeTypeBinding(getContext(), *innerInputBinding);

  auto arrValue = getCastedValue(adaptor.getArray(), *inputBinding, rewriter);
  auto one = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), 1);
  auto zero = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), 0);

  auto len = rewriter.create<GetArrayLenOp>(op.getLoc(), arrValue);

  auto maybeSuperCoerce = [&](Value v, Type t) -> Value {
    if (v.getType() == t) {
      return v;
    }
    return rewriter.create<SuperCoerceOp>(v.getLoc(), t, v);
  };

  auto loop = rewriter.create<mlir::scf::ForOp>(
      op.getLoc(), zero, len->getResult(0), one, mlir::ValueRange(*init),
      [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange args) {
    arrayFrame->setInductionVar(iv);
    mlir::Value lhs = maybeSuperCoerce(
        builder.create<ReadArrayOp>(loc, itType, arrValue, mlir::ValueRange(iv)),
        constructorType.getInput(0)
    );

    mlir::Value rhs = maybeSuperCoerce(args[0], constructorType.getInput(1));
    auto accResult = ctorBuilder->build(builder, adaptor.getType().getLoc(), {lhs, rhs});
    builder.create<mlir::scf::YieldOp>(loc, maybeSuperCoerce(accResult, init->getType()));
  }
  );

  rewriter.replaceOp(op, loop);
  loop->setAttr("original_op", rewriter.getStringAttr("reduce"));

  return mlir::success();
}

Value createNthCond(unsigned int idx, Value selector, OpBuilder &rewriter) {
  auto val = ComponentType::Val(rewriter.getContext());
  // Load the selector value from the array
  auto nth = rewriter.create<LitValOp>(selector.getLoc(), val, idx);
  auto item = rewriter.create<ReadArrayOp>(selector.getLoc(), val, selector, ValueRange(nth));

  // Check if the value is equal to 1 (by converting it into a boolean)
  return rewriter.create<ValToI1Op>(selector.getLoc(), item);
}

/// Inlines a switch arm region. The region must have only 1 block.
void inlineRegion(
    Region *region, Block &dest, Block::iterator it, ConversionPatternRewriter &rewriter
) {
  assert(region->getBlocks().size() == 1);
  rewriter.inlineBlockBefore(&region->front(), &dest, it);
}

/// Builds an if-then-else chain with each region of the switch op.
void buildIfThenElseChain(
    RegionRange::iterator region_begin, RegionRange::iterator region_end,
    ValueRange::iterator conds, Block &dest, Block::iterator destIt,
    ConversionPatternRewriter &rewriter, Type retType
) {
  Value cond = *conds;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(&dest, destIt);
  if (std::next(region_begin) == region_end) {
    rewriter.create<AssertOp>(cond.getLoc(), cond);
    inlineRegion(*region_begin, dest, rewriter.getInsertionPoint(), rewriter);
    return;
  }

  auto ifOp = rewriter.create<scf::IfOp>(cond.getLoc(), retType, cond, true, true);
  inlineRegion(
      *region_begin, ifOp.getThenRegion().front(), ifOp.getThenRegion().front().end(), rewriter
  );
  buildIfThenElseChain(
      std::next(region_begin), region_end, std::next(conds), ifOp.getElseRegion().front(),
      ifOp.getElseRegion().front().end(), rewriter, retType
  );
  rewriter.create<scf::YieldOp>(ifOp.getLoc(), ifOp.getResults());
}

LogicalResult ZhlSwitchLowering::matchAndRewrite(
    Zhl::SwitchOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }
  auto arrType =
      ComponentType::Array(getContext(), ComponentType::Val(getContext()), op.getNumRegions());
  auto selector = getCastedValue(adaptor.getSelector(), rewriter, arrType);
  if (failed(selector)) {
    return op->emitOpError() << "failed to type check selector";
  }
  SmallVector<Value> conds;
  conds.reserve(op.getNumRegions());
  for (unsigned int n = 0; n < op.getNumRegions(); n++) {
    conds.push_back(createNthCond(n, *selector, rewriter));
  }
  ValueRange condsRange(conds);

  auto retType = materializeTypeBinding(getContext(), *binding);
  RegionRange regions = op.getRegions();
  auto execRegion = rewriter.replaceOpWithNewOp<scf::ExecuteRegionOp>(op, retType);

  auto &block = execRegion.getRegion().emplaceBlock();
  buildIfThenElseChain(
      regions.begin(), regions.end(), condsRange.begin(), block, block.end(), rewriter, retType
  );

  return success();
}

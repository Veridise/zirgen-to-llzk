#include "zklang/Passes/ConvertZhlToZml/Patterns.h"
#include "zklang/Dialect/ZHL/Typing/TypeBindings.h"
#include "zklang/Dialect/ZML/IR/Builder.h"
#include "zklang/Dialect/ZML/IR/OpInterfaces.h"
#include "zklang/Dialect/ZML/IR/Ops.h"
#include "zklang/Dialect/ZML/IR/Types.h"
#include "zklang/Dialect/ZML/Typing/Materialize.h"
#include "zklang/Dialect/ZML/Typing/ZMIRTypeConverter.h"
#include "zklang/Passes/ConvertZhlToZml/Helpers.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Index/IR/IndexAttrs.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <vector>

using namespace zirgen;
using namespace zkc;
using namespace zhl;
using namespace mlir;

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

  auto val = Zmir::materializeTypeBinding(getContext(), *binding);
  rewriter.replaceOpWithNewOp<Zmir::LitValOp>(op, val, adaptor.getValue());
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
  auto val = Zmir::materializeTypeBinding(getContext(), *binding);
  rewriter.replaceOpWithNewOp<Zmir::LitStrOp>(op, val, adaptor.getValue());
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

  rewriter.replaceOpWithNewOp<Zmir::NopOp>(op, arg.getType(), arg);
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlConstructLowering
///////////////////////////////////////////////////////////

/// Finds the definition of the callee component. If the
/// component was defined before the current operation wrt the physical order of
/// the file then its defined by a ZML ComponentInterface op, if it hasn't been
/// converted yet it is still a ZHL Component op. If the name could not be found
/// in either form returns nullptr.
mlir::Operation *findCallee(StringRef name, mlir::ModuleOp root) {
  auto calleeComp = root.lookupSymbol<Zmir::ComponentInterface>(name);
  if (calleeComp) {
    return calleeComp;
  }

  // Zhl Component ops don't declare its symbols in the symbol table
  for (auto zhlOp : root.getOps<Zhl::ComponentOp>()) {
    if (zhlOp.getName() == name) {
      return zhlOp;
    }
  }
  return nullptr;
}

bool calleeIsBuiltin(mlir::Operation *op) {
  if (auto zmlOp = mlir::dyn_cast<Zmir::ComponentInterface>(op)) {
    return zmlOp.getBuiltin();
  }
  return false;
}

mlir::LogicalResult ZhlConstructLowering::matchAndRewrite(
    Zhl::ConstructOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op.emitOpError("failed to type check");
  }

  auto callerComp = op->getParentOfType<Zmir::ComponentOp>();
  assert(callerComp);
  auto *calleeComp = findCallee(binding->getName(), op->getParentOfType<mlir::ModuleOp>());
  if (!calleeComp) {
    return op->emitError() << "could not find component with name " << binding->getName();
  }

  /*auto constructorTypes = calleeComp.getBodyFunc().getFunctionType().getInputs();*/
  auto constructorType = Zmir::materializeTypeBindingConstructor(rewriter, *binding);
  {
    auto constructorTypes = constructorType.getInputs();
    bool isVariadic =
        !constructorTypes.empty() && mlir::isa<Zmir::VarArgsType>(constructorTypes.back());
    // Depending if it's variadic or not the message changes a bit.
    std::string expectingNArgsMsg = isVariadic ? ", was expecting at least " : ", was expecting ";

    // Can be constructorTypes.size() - 1 as the variadic args can be empty
    bool isInRangeVariadic = isVariadic && (adaptor.getArgs().size() >= (constructorTypes.size() - 1));
    bool isInRangeNonVariadic = !isVariadic && (adaptor.getArgs().size() == constructorTypes.size());

    if (!isInRangeVariadic && !isInRangeNonVariadic) {
      return op->emitOpError()
          .append(
              "incorrect number of arguments for component ", binding->getName(), expectingNArgsMsg,
              constructorTypes.size(), " arguments and got ", adaptor.getArgs().size()
          )
          .attachNote(calleeComp->getLoc())
          .append("component declared here");
    }
  }

  std::vector<mlir::Value> preparedArguments;
  prepareArguments(
      adaptor.getArgs(), constructorType.getInputs(), op->getLoc(), rewriter, preparedArguments
  );

  auto funcPtr = rewriter.create<Zmir::ConstructorRefOp>(
      op.getLoc(), mlir::SymbolRefAttr::get(rewriter.getStringAttr(binding->getName())),
      constructorType, calleeIsBuiltin(calleeComp)
  );
  auto call = rewriter.create<mlir::func::CallIndirectOp>(op->getLoc(), funcPtr, preparedArguments);

  auto result = storeValueInTemporary(
      op.getLoc(), callerComp, constructorType.getResult(0), call.getResult(0), rewriter
  );

  rewriter.replaceOp(op, result);

  rewriter.create<Zmir::ConstrainCallOp>(
      op->getLoc(), result->getResult(0), ValueRange(preparedArguments)
  );
  return mlir::success();
}

void ZhlConstructLowering::prepareArguments(
    mlir::ValueRange args, mlir::ArrayRef<mlir::Type> constructorTypes, mlir::Location loc,
    mlir::ConversionPatternRewriter &rewriter, std::vector<mlir::Value> &preparedArgs
) const {
  bool isVariadic =
      !constructorTypes.empty() && mlir::isa<Zmir::VarArgsType>(constructorTypes.back());

  preparedArgs.clear();

  auto argsEnd = isVariadic ? args.begin() + (args.size() - constructorTypes.size()) : args.end();
  std::transform(
      args.begin(), argsEnd, constructorTypes.begin(), std::back_inserter(preparedArgs),
      [&](mlir::Value v, mlir::Type t) { return prepareArgument(v, t, loc, rewriter); }
  );
  if (isVariadic) {
    std::vector<mlir::Value> vargs;
    auto varType = mlir::dyn_cast<Zmir::VarArgsType>(constructorTypes.back());
    assert(varType && "expecting a var args type");

    std::transform(argsEnd, args.end(), std::back_inserter(vargs), [&](mlir::Value v) {
      return prepareArgument(v, varType.getInner(), loc, rewriter);
    });

    auto va = rewriter.create<Zmir::VarArgsOp>(loc, constructorTypes.back(), vargs);
    preparedArgs.push_back(va);

    // If the varargs value is empty, then prepared args will be smaller than constructorTypes
    // by one, otherwise it can be arbitrarily larger than the constructorTypes.
    assert(preparedArgs.size() >= constructorTypes.size() - 1 && "incorrect number of arguments");
  } else {
    // Without varargs, the number of arguments should match exactly
    assert(preparedArgs.size() == constructorTypes.size() && "incorrect number of arguments");
  }
}

mlir::Value ZhlConstructLowering::prepareArgument(
    mlir::Value arg, mlir::Type expectedType, mlir::Location loc,
    mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(arg);
  auto type = expectedType;

  if (mlir::succeeded(binding)) {
    type = Zmir::materializeTypeBinding(rewriter.getContext(), *binding);
  }
  if (arg.getType() == type) {
    return arg;
  }
  auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, type, arg);
  if (expectedType == cast.getResult(0).getType()) {
    return cast.getResult(0);
  }
  return rewriter.create<Zmir::SuperCoerceOp>(loc, expectedType, cast.getResult(0));
}

mlir::FailureOr<mlir::Type> ZhlConstructLowering::getTypeFromName(mlir::StringRef name) const {
  // Simple algorithm for now
  if (name == "Val") {
    return Zmir::ValType::get(getContext());
  } else if (name == "String") {
    return Zmir::StringType::get(getContext());
  } else {
    return mlir::failure(); // For now
  }
}

mlir::LogicalResult ZhlConstrainLowering::matchAndRewrite(
    Zhl::ConstraintOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {

  auto lhsBinding = getType(op.getLhs());
  if (mlir::failed(lhsBinding)) {
    return op->emitOpError() << "failed to type check lhs";
  }
  auto rhsBinding = getType(op.getRhs());
  if (mlir::failed(rhsBinding)) {
    return op->emitOpError() << "failed to type check rhs";
  }

  mlir::Value lhsValue = adaptor.getLhs();
  mlir::Type lhsTarget = Zmir::materializeTypeBinding(getContext(), *lhsBinding);
  if (lhsValue.getType() != lhsTarget) {
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), lhsTarget, lhsValue);
    lhsValue = cast.getResult(0);
  }
  // Coerce to Val if necessary
  if (lhsValue.getType() != Zmir::ComponentType::Val(getContext())) {
    lhsValue = rewriter.create<Zmir::SuperCoerceOp>(
        op.getLoc(), Zmir::ComponentType::Val(getContext()), lhsValue
    );
  }
  mlir::Value rhsValue = adaptor.getRhs();
  mlir::Type rhsTarget = Zmir::materializeTypeBinding(getContext(), *rhsBinding);
  if (rhsValue.getType() != rhsTarget) {
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), rhsTarget, rhsValue);
    rhsValue = cast.getResult(0);
  }
  // Coerce to Val if necessary
  if (rhsValue.getType() != Zmir::ComponentType::Val(getContext())) {
    rhsValue = rewriter.create<Zmir::SuperCoerceOp>(
        op.getLoc(), Zmir::ComponentType::Val(getContext()), rhsValue
    );
  }
  rewriter.replaceOpWithNewOp<Zmir::ConstrainOp>(op, lhsValue, rhsValue);
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
  rewriter.replaceOpWithNewOp<Zmir::NopOp>(
      op, mlir::TypeRange({Zmir::materializeTypeBinding(getContext(), *binding)}),
      mlir::ValueRange()
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
  auto type = Zmir::materializeTypeBinding(getContext(), *binding);
  if (binding->isGenericParam() && binding->getSuperType().isVal()) {
    rewriter.replaceOpWithNewOp<Zmir::LoadValParamOp>(
        op, Zmir::ComponentType::Val(getContext()), op.getNameAttr()
    );
  } else {
    rewriter.replaceOpWithNewOp<Zmir::NopOp>(op, mlir::TypeRange({type}), mlir::ValueRange());
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
  rewriter.replaceOpWithNewOp<Zmir::NopOp>(
      op, mlir::TypeRange({Zmir::materializeTypeBinding(getContext(), *binding)}),
      mlir::ValueRange()
  );
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlDeclarationRemoval
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlDeclarationRemoval::matchAndRewrite(
    zirgen::Zhl::DeclarationOp op, OpAdaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<Zmir::NopOp>(
      op, mlir::TypeRange({Zmir::ComponentType::Component(rewriter.getContext())}),
      mlir::ValueRange()
  );
  return mlir::success();
}

inline bool opConstructsComponent(mlir::Operation *op) {
  // TODO: Add the rest of the ops: array constructors, for loops, etc
  return mlir::isa<func::CallIndirectOp>(op);
}

/// Tries to get an operation from the value written into the field.
/// If it is then adds an attribute into the operation with the name of the
/// field.
/// FIXME: This is a hacky solution to a problem that probably needs some static
/// analysis
void maybeAnnotateConstructorCallWithField(Zmir::WriteFieldOp op, mlir::Value value) {
  auto valueOp = value.getDefiningOp();
  if (!valueOp) {
    return; // The value doesn't come from an op
  }
  if (opConstructsComponent(valueOp)) {
    valueOp->setAttr("writes_into", mlir::StringAttr::get(op.getContext(), op.getFieldName()));
    return; // We are done
  }

  if (valueOp->getNumOperands() != 1) {
    return;
  }
  // If the operation has only one operand try recursively
  maybeAnnotateConstructorCallWithField(op, valueOp->getOperand(0));
}

///////////////////////////////////////////////////////////
/// ZhlDefinitionLowering
///////////////////////////////////////////////////////////

void createField(
    Zmir::ComponentInterface comp, mlir::StringRef name, mlir::Type type,
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc
) {
  assert(comp);

  mlir::OpBuilder::InsertionGuard guard(rewriter);

  rewriter.setInsertionPointToStart(&comp.getRegion().front());
  rewriter.create<Zmir::FieldDefOp>(loc, name, mlir::dyn_cast<Zmir::ComponentType>(type));
}

mlir::LogicalResult ZhlDefineLowering::matchAndRewrite(
    zirgen::Zhl::DefinitionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }

  Zhl::DeclarationOp declr = op.getDeclaration().getDefiningOp<Zhl::DeclarationOp>();
  if (!declr) {
    return op.emitOpError("does not depend on a declaration");
  }
  auto name = declr.getMemberAttr();

  auto comp = op->getParentOfType<Zmir::ComponentOp>();
  assert(comp);

  mlir::Value value = adaptor.getDefinition();
  mlir::Type target = Zmir::materializeTypeBinding(getContext(), *binding);
  if (value.getType() != target) {
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), target, value);
    value = cast.getResult(0);
  }

  auto self = rewriter.create<Zmir::GetSelfOp>(op.getLoc(), comp.getType());
  auto writeOp = rewriter.replaceOpWithNewOp<Zmir::WriteFieldOp>(op, self, name, value);
  maybeAnnotateConstructorCallWithField(writeOp, value);
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlSuperLowering
///////////////////////////////////////////////////////////

/// A best-effort check of known externs to the type
/// they return.
/*mlir::FailureOr<mlir::Type> deduceTypeOfKnownExtern(mlir::StringRef name, mlir::MLIRContext *ctx)
 * {*/
/*  if (name == "Log") {*/
/*    return Zmir::ComponentType::get(ctx, "Component");*/
/*  }*/
/**/
/*  return mlir::failure();*/
/*}*/

mlir::LogicalResult ZhlSuperLoweringInFunc::matchAndRewrite(
    zirgen::Zhl::SuperOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }
  if (!mlir::isa<mlir::func::FuncOp>(op->getParentOp())) {
    return mlir::failure();
  }
  auto comp = op->getParentOfType<Zmir::ComponentInterface>();
  assert(comp);
  auto self = rewriter.create<Zmir::GetSelfOp>(op.getLoc(), comp.getType());

  mlir::Value value = adaptor.getValue();
  mlir::Type target = Zmir::materializeTypeBinding(getContext(), *binding);
  if (value.getType() != target) {
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), target, value);
    value = cast.getResult(0);
  }

  /*createField(comp, "$super", type, rewriter, op.getLoc());*/
  auto writeOp = rewriter.replaceOpWithNewOp<Zmir::WriteFieldOp>(op, self, "$super", value);
  maybeAnnotateConstructorCallWithField(writeOp, adaptor.getValue());

  // Find the prologue and join it to the current block
  auto *prologue = &comp.getBodyFunc().getRegion().back();
  auto nop =
      rewriter.create<Zmir::NopOp>(rewriter.getUnknownLoc(), mlir::TypeRange(), mlir::ValueRange());
  rewriter.inlineBlockBefore(prologue, nop.getOperation());
  rewriter.eraseOp(nop);
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
    Zmir::ComponentInterface op, llvm::StringRef name, mlir::FunctionType type,
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc
) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op.getBodyFunc());
  auto attrs = externFuncAttrs(rewriter);

  return rewriter.create<mlir::func::FuncOp>(loc, name, type, attrs);
}

inline mlir::FailureOr<mlir::Type>
lookupComponentNameIn(mlir::Operation *op, mlir::StringRef name) {
  assert(op && "expected a non null operation");
  auto sym =
      mlir::SymbolTable::lookupNearestSymbolFrom(op, mlir::StringAttr::get(op->getContext(), name));
  if (!sym) {
    return mlir::failure();
  }
  if (auto comp = mlir::dyn_cast<Zmir::ComponentOp>(sym)) {
    return comp.getType();
  }
  return mlir::failure();
  ;
}

inline mlir::FailureOr<mlir::Type> getTypeFromName(Zhl::GlobalOp typeNameOp) {
  auto name = typeNameOp.getName();
  if (name == "Val") {
    return Zmir::ValType::get(typeNameOp.getContext());
  } else if (name == "String") {
    return Zmir::StringType::get(typeNameOp.getContext());
  } else {
    auto lookupFromOp = lookupComponentNameIn(typeNameOp, name);
    if (mlir::succeeded(lookupFromOp)) {
      return lookupFromOp;
    }

    auto lookupFromMod = lookupComponentNameIn(typeNameOp->getParentOfType<mlir::ModuleOp>(), name);
    if (mlir::succeeded(lookupFromMod)) {
      return lookupFromMod;
    }

    return typeNameOp.emitError() << "type " << name << " not found";
  }
}

mlir::LogicalResult ZhlExternLowering::matchAndRewrite(
    Zhl::ExternOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }
  auto comp = op->getParentOfType<Zmir::ComponentOp>();
  if (!comp) {
    return mlir::failure();
  }

  std::vector<mlir::Type> argBindings;
  for (auto arg : op.getArgs()) {
    auto argBinding = getType(arg);
    if (mlir::failed(argBinding)) {
      return op->emitOpError() << "failed to type check argument #" << argBindings.size();
    }
    argBindings.push_back(Zmir::materializeTypeBinding(getContext(), *argBinding));
  }

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

  auto retType = Zmir::materializeTypeBinding(getContext(), *binding);

  // Extern ops are wrapped around a component by the AST->ZHL step and have the same inputs as the
  // component.
  auto funcType =
      rewriter.getFunctionType(comp.getBodyFunc().getFunctionType().getInputs(), {retType});
  std::string externName(op.getName().str());
  externName += "$$extern";
  auto externDeclrResult = createExternFunc(comp, externName, funcType, rewriter, op.getLoc());
  if (mlir::failed(externDeclrResult)) {
    return mlir::failure();
  }

  rewriter.replaceOpWithNewOp<mlir::func::CallOp>(op, *externDeclrResult, mlir::ValueRange(args));

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
  auto materializedType = Zmir::materializeTypeBinding(getContext(), *originalComp);
  auto compType = mlir::dyn_cast<Zmir::ComponentType>(materializedType);
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
    compType = mlir::dyn_cast<Zmir::ComponentType>(superType);
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
  auto bindingType = Zmir::materializeTypeBinding(getContext(), *binding);
  if (*fieldType != bindingType) {
    return op->emitError() << "type mismatch, was expecting " << bindingType << " but field "
                           << adaptor.getMember() << " is of type " << *fieldType;
  }

  // Coerce to the type in the chain that defines the accessed member
  comp = rewriter.create<Zmir::SuperCoerceOp>(op.getLoc(), compType, comp);

  rewriter.replaceOpWithNewOp<Zmir::ReadFieldOp>(op, bindingType, comp, adaptor.getMember());
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

  auto arrayVal = adaptor.getArray();
  auto arrayType = Zmir::materializeTypeBinding(getContext(), *arrayBinding);
  if (arrayVal.getType() != arrayType) {
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), arrayType, arrayVal);
    arrayVal = cast.getResult(0);
  }
  auto elementVal = adaptor.getElement();
  auto elementType = Zmir::materializeTypeBinding(getContext(), *elementBinding);
  if (elementVal.getType() != elementType) {
    auto cast =
        rewriter.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), elementType, elementVal);
    elementVal = cast.getResult(0);
  }
  rewriter.replaceOpWithNewOp<Zmir::ReadArrayOp>(
      op, Zmir::materializeTypeBinding(getContext(), *binding), arrayVal, elementVal
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
  auto arrayType = Zmir::materializeTypeBinding(getContext(), *binding);
  auto elementTypeBinding = binding->getArrayElement([&]() { return op->emitOpError(); });
  if (mlir::failed(elementTypeBinding)) {
    return mlir::failure();
  }
  auto elementType = Zmir::materializeTypeBinding(getContext(), *elementTypeBinding);

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
    rewriter.replaceOpWithNewOp<Zmir::AllocArrayOp>(
        op, Zmir::UnboundedArrayType::get(
                rewriter.getContext(), Zmir::PendingType::get(rewriter.getContext())
            )
    );
    return mlir::success();
  }

  llvm::SmallVector<mlir::Value> args;

  std::transform(
      adaptor.getElements().begin(), adaptor.getElements().end(), argBindings.begin(),
      std::back_inserter(args),
      [&](auto element, auto eltBinding) -> Value {
    auto bindingType = Zmir::materializeTypeBinding(getContext(), *eltBinding);
    if (element.getType() == bindingType) {
      return element;
    }

    auto cast =
        rewriter.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), bindingType, element);
    if (cast.getResult(0).getType() == elementType) {
      return cast.getResult(0);
    }
    return rewriter.create<Zmir::SuperCoerceOp>(op.getLoc(), elementType, cast.getResult(0));
  }
  );
  auto arr = rewriter.create<Zmir::NewArrayOp>(op.getLoc(), arrayType, args);
  rewriter.replaceOp(
      op, storeValueInTemporary(
              op.getLoc(), op->getParentOfType<Zmir::ComponentOp>(), arrayType, arr, rewriter
          )
  );
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

  Zmir::ComponentBuilder builder;
  auto genericNames = name->getGenericParamNames();
  builder.name(name->getName())
      .location(op->getLoc())
      .attrs(op->getAttrs())
      .typeParams(genericNames)
      .constructor(
          Zmir::materializeTypeBindingConstructor(rewriter, *name),
          name->getConstructorParamLocations()
      )
      .takeRegion(&op.getRegion());
  for (auto &[fieldName, binding] : name->getMembers()) {
    if (!binding.has_value()) {
      return op->emitOpError() << "failed to type check component member '" << fieldName << "'";
    }
    builder.field(
        fieldName, Zmir::materializeTypeBinding(getContext(), *binding), binding->getLocation()
    );
  }
  auto &super = name->getSuperType();
  builder.field("$super", Zmir::materializeTypeBinding(getContext(), super));
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
  auto startBinding = getType(op.getStart());
  if (mlir::failed(startBinding)) {
    return op->emitOpError() << "failed to type check start";
  }
  auto endBinding = getType(op.getEnd());
  if (mlir::failed(endBinding)) {
    return op->emitOpError() << "failed to type check end";
  }
  auto type = Zmir::materializeTypeBinding(getContext(), *binding);
  assert(binding->isArray());
  auto innerBinding = binding->getArrayElement([&]() { return op->emitError(); });
  if (mlir::failed(innerBinding)) {
    return mlir::failure();
  }
  auto innerType = Zmir::materializeTypeBinding(getContext(), *innerBinding);
  auto startType = Zmir::materializeTypeBinding(getContext(), *startBinding);
  Value startVal = adaptor.getStart();
  if (startType != startVal.getType()) {
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
        startVal.getLoc(), TypeRange(startType), ValueRange(startVal)
    );
    startVal = cast.getResult(0);
  }
  if (startVal.getType() != Zmir::ComponentType::Val(getContext())) {
    startVal = rewriter.create<Zmir::SuperCoerceOp>(
        startVal.getLoc(), Zmir::ComponentType::Val(getContext()), startVal
    );
  }
  auto endType = Zmir::materializeTypeBinding(getContext(), *endBinding);
  Value endVal = adaptor.getEnd();
  if (endType != endVal.getType()) {
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
        endVal.getLoc(), TypeRange(endType), ValueRange(endVal)
    );
    endVal = cast.getResult(0);
  }
  if (endVal.getType() != Zmir::ComponentType::Val(getContext())) {
    endVal = rewriter.create<Zmir::SuperCoerceOp>(
        endVal.getLoc(), Zmir::ComponentType::Val(getContext()), endVal
    );
  }
  auto arrAlloc = rewriter.replaceOpWithNewOp<Zmir::AllocArrayOp>(op, type);
  // Create a for loop op using the operands as bounds
  auto one = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), 1);
  auto start = rewriter.create<Zmir::ValToIndexOp>(op.getStart().getLoc(), startVal);
  auto end = rewriter.create<Zmir::ValToIndexOp>(op.getEnd().getLoc(), endVal);
  rewriter.create<mlir::scf::ForOp>(
      op.getLoc(), start, end, one, mlir::ValueRange(),
      [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange) {
    auto conv = builder.create<Zmir::IndexToValOp>(loc, innerType, iv);
    builder.create<Zmir::WriteArrayOp>(loc, arrAlloc, iv, conv);
    builder.create<mlir::scf::YieldOp>(loc);
  }
  );

  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlRangeOpLowering
///////////////////////////////////////////////////////////

inline mlir::Type tryGetArrayInnerType(mlir::Type type) {
  if (auto a = mlir::dyn_cast<Zmir::ArrayType>(type)) {
    return a.getInnerType();
  }
  return Zmir::PendingType::get(type.getContext());
}

inline Zmir::ArrayType resultArrayType(
    mlir::Type inner, int64_t size, mlir::MLIRContext *ctx,
    mlir::ConversionPatternRewriter &rewriter
) {
  if (size >= 0) {
    return Zmir::BoundedArrayType::get(ctx, inner, rewriter.getIndexAttr(size));
  }
  return Zmir::UnboundedArrayType::get(ctx, inner);
}

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

  auto itType = Zmir::materializeTypeBinding(getContext(), *innerInputBinding);
  auto outputType = Zmir::materializeTypeBinding(getContext(), *binding);

  auto arrValue = adaptor.getArray();

  auto arrAlloc = rewriter.create<Zmir::AllocArrayOp>(op.getLoc(), outputType);
  auto one = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), 1);
  auto zero = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), 0);
  auto len = rewriter.create<Zmir::GetArrayLenOp>(op.getLoc(), adaptor.getArray());

  auto loop = rewriter.create<mlir::scf::ForOp>(
      op.getLoc(), zero, len->getResult(0), one, mlir::ValueRange(arrAlloc),
      [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange args) {
    auto itVal = builder.create<Zmir::ReadArrayOp>(loc, itType, arrValue, mlir::ValueRange(iv));
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

  auto type = Zmir::materializeTypeBinding(getContext(), *binding);
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

  auto iv = loopOp.getInductionVar();
  auto arr = loopOp.getRegionIterArgs().front();

  rewriter.create<Zmir::WriteArrayOp>(op.getLoc(), arr, iv, adaptor.getValue());
  rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, loopOp.getRegionIterArgs());

  return mlir::success();
}

mlir::LogicalResult ZhlSuperLoweringInBlock::matchAndRewrite(
    Zhl::SuperOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto parent = op->getParentOp();
  if (!parent || !mlir::isa<mlir::scf::ExecuteRegionOp>(parent)) {
    return mlir::failure();
  }
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }

  auto value = adaptor.getValue();
  auto type = Zmir::materializeTypeBinding(getContext(), *binding);
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

  auto value = adaptor.getValue();
  auto type = Zmir::materializeTypeBinding(getContext(), *binding);
  if (value.getType() != type) {
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), type, value);
    value = cast.getResult(0);
  }

  rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, value);
  return mlir::success();
}

mlir::LogicalResult ZhlReduceLowering::matchAndRewrite(
    Zhl::ReduceOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (mlir::failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }

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

  auto constructorType = Zmir::materializeTypeBindingConstructor(rewriter, *accBinding);
  assert(constructorType);
  if (constructorType.getInputs().size() != 2) {
    return op->emitOpError() << "was expecting a constructor with two arguments but got "
                             << constructorType.getInputs().size() << " arguments";
  }

  auto itType = Zmir::materializeTypeBinding(getContext(), *innerInputBinding);

  auto arrValue = adaptor.getArray();

  auto one = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), 1);
  auto zero = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), 0);
  auto len = rewriter.create<Zmir::GetArrayLenOp>(op.getLoc(), adaptor.getArray());

  auto loop = rewriter.create<mlir::scf::ForOp>(
      op.getLoc(), zero, len->getResult(0), one, mlir::ValueRange(adaptor.getInit()),
      [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange args) {
    mlir::Value lhs =
        builder.create<Zmir::ReadArrayOp>(loc, itType, arrValue, mlir::ValueRange(iv));
    if (lhs.getType() != constructorType.getInputs()[0]) {
      lhs = builder.create<Zmir::SuperCoerceOp>(loc, constructorType.getInputs()[0], lhs);
    }
    mlir::Value rhs = args[0];
    if (rhs.getType() != constructorType.getInputs()[1]) {
      rhs = builder.create<Zmir::SuperCoerceOp>(loc, constructorType.getInputs()[1], rhs);
    }
    auto ref = builder.create<Zmir::ConstructorRefOp>(loc, constructorType, accBinding->getName());
    auto call = builder.create<mlir::func::CallIndirectOp>(loc, ref, mlir::ValueRange({lhs, rhs}));
    mlir::Value acc = call.getResult(0);
    if (acc.getType() != adaptor.getInit().getType()) {
      acc = builder.create<Zmir::SuperCoerceOp>(loc, adaptor.getInit().getType(), acc);
    }
    builder.create<mlir::scf::YieldOp>(loc, acc);
  }
  );

  rewriter.replaceOp(op, loop);
  loop->setAttr("original_op", rewriter.getStringAttr("reduce"));

  return mlir::success();
}

/// Builds an if-then-else chain with each region of the switch op.
/// The regions are inserted in the opposite order because the condition
/// used in the `scf.if` op is the inverse of the condition used in ZIR.
Operation *buildIfThenElseChain(
    RegionRange::iterator region_begin, RegionRange::iterator region_end,
    ConversionPatternRewriter &rewriter, int idx, Value selector, Type retType
) {
  auto val = Zmir::ComponentType::Val(rewriter.getContext());

  // If we reach the end we only generate the final assert.
  if (region_begin == region_end) {
    auto zero = rewriter.create<Zmir::LitValOp>(rewriter.getUnknownLoc(), val, 0);
    return rewriter.create<Zmir::AssertOp>(rewriter.getUnknownLoc(), retType, zero);
  }
  auto region = *region_begin;

  // Test if the nth element of the selector is zero.
  auto nth = rewriter.create<Zmir::LitValOp>(rewriter.getUnknownLoc(), val, idx);
  auto item = rewriter.create<Zmir::ReadArrayOp>(region->getLoc(), val, selector, ValueRange(nth));
  auto isz = rewriter.create<Zmir::IsZeroOp>(region->getLoc(), val, item);
  auto toBool = rewriter.create<Zmir::ValToI1Op>(region->getLoc(), isz);
  auto ifOp = rewriter.create<scf::IfOp>(region->getLoc(), retType, toBool, true, true);

  // If it's not zero then we execute the code of the nth region that has been copied to the else
  // branch.
  {
    auto &elseBlock = ifOp.getElseRegion().front();
    assert(region->getBlocks().size() == 1);
    rewriter.inlineBlockBefore(&region->front(), &elseBlock, elseBlock.end());
  }

  // If it's zero then we execute the then branch that recursively contains the pending regions to
  // check.
  {
    auto &thenBlock = ifOp.getThenRegion().front();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&thenBlock);

    ++region_begin;
    auto chainOp =
        buildIfThenElseChain(region_begin, region_end, rewriter, idx + 1, selector, retType);
    assert(chainOp->getNumResults() == 1);
    Value chain = chainOp->getResult(0);
    if (chain.getType() != retType) {
      chain = rewriter.create<Zmir::SuperCoerceOp>(chainOp->getLoc(), retType, chain);
    }
    rewriter.create<scf::YieldOp>(chainOp->getLoc(), chain);
  }
  return ifOp;
}

LogicalResult ZhlSwitchLowering::matchAndRewrite(
    Zhl::SwitchOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
) const {
  auto binding = getType(op);
  if (failed(binding)) {
    return op->emitOpError() << "failed to type check";
  }
  auto arrType = Zmir::ComponentType::Array(
      getContext(), Zmir::ComponentType::Val(getContext()), op.getNumRegions()
  );
  auto selector = adaptor.getSelector();
  auto selectorBinding = getType(op.getSelector());
  if (failed(selectorBinding)) {
    return op->emitOpError() << "failed to type check selector";
  }
  auto selectorType = Zmir::materializeTypeBinding(getContext(), *selectorBinding);
  if (selector.getType() != selectorType) {
    selector =
        rewriter.create<Zmir::SuperCoerceOp>(op.getSelector().getLoc(), selectorType, selector);
  }
  if (selector.getType() != arrType) {
    selector = rewriter.create<Zmir::SuperCoerceOp>(op.getSelector().getLoc(), arrType, selector);
  }

  auto retType = Zmir::materializeTypeBinding(getContext(), *binding);
  RegionRange regions = op.getRegions();
  auto chain = buildIfThenElseChain(regions.begin(), regions.end(), rewriter, 0, selector, retType);

  rewriter.replaceOp(op.getOperation(), chain);

  return success();
}

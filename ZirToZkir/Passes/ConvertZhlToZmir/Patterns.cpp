#include "Patterns.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include "ZirToZkir/Dialect/ZMIR/Typing/ZMIRTypeConverter.h"
#include "ZirToZkir/Passes/ConvertZhlToZmir/Helpers.h"
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
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>
#include <vector>

using namespace zirgen;
using namespace zkc;

///////////////////////////////////////////////////////////
/// Cast folding
///////////////////////////////////////////////////////////

mlir::LogicalResult FoldUnrealizedCasts::matchAndRewrite(
    mlir::UnrealizedConversionCastOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter
) const {

  if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) {
    return mlir::failure();
  }

  if (!getTypeConverter()->isLegal(op.getOutputs()[0].getType())) {
    return mlir::failure();
  }

  auto parent = adaptor.getInputs()[0].getDefiningOp();
  if (!parent) {
    return mlir::failure();
  }
  auto parentCast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(parent);
  if (!parentCast) {
    return mlir::failure();
  }

  if (parentCast.getInputs().size() != 1 || parentCast.getOutputs().size() != 1) {
    return mlir::failure();
  }
  if (!getTypeConverter()->isLegal(parentCast.getInputs()[0].getType())) {
    return mlir::failure();
  }

  rewriter.replaceAllUsesWith(op.getOutputs()[0], parentCast.getInputs()[0]);
  rewriter.eraseOp(op);

  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlLiteralLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlLiteralLowering::matchAndRewrite(
    Zhl::LiteralOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  Zmir::ValType val = Zmir::ValType::get(getContext());
  rewriter.replaceOpWithNewOp<Zmir::LitValOp>(op, val, adaptor.getValue());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlLiteralStrLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlLiteralStrLowering::matchAndRewrite(
    Zhl::StringOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto val = Zmir::StringType::get(getContext());
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
  auto conv = getTypeConverter()->materializeSourceConversion(
      rewriter, op.getLoc(), Zhl::ExprType::get(getContext()), {arg}
  );

  rewriter.replaceAllUsesWith(op, conv);
  rewriter.eraseOp(op);
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlConstructLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlConstructLowering::matchAndRewrite(
    Zhl::ConstructOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  Zhl::GlobalOp typeNameOp = op.getType().getDefiningOp<Zhl::GlobalOp>();
  if (!typeNameOp) {
    return op.emitError("constructed type is not declared by a global");
  }

  auto calleeComp = op->getParentOfType<mlir::ModuleOp>().lookupSymbol<Zmir::ComponentInterface>(
      typeNameOp.getNameAttr()
  );
  if (!calleeComp) {
    return op->emitError() << "could not find component with name " << typeNameOp.getNameAttr();
  }

  auto constructorTypes = calleeComp.getBodyFunc().getFunctionType().getInputs();
  {
    bool isVariadic =
        !constructorTypes.empty() && mlir::isa<Zmir::VarArgsType>(constructorTypes.back());
    // Depending if it's variadic or not the message changes a bit.
    std::string expectingNArgsMsg = isVariadic ? " was expecting at least " : " was expecting ";

    if (adaptor.getArgs().size() < constructorTypes.size() ||
        (!isVariadic && adaptor.getArgs().size() > constructorTypes.size())) {
      return op->emitOpError()
          .append(
              "incorrect number of arguments for component ", typeNameOp.getNameAttr(),
              expectingNArgsMsg, constructorTypes.size(), " arguments and got ",
              adaptor.getArgs().size()
          )
          .attachNote(calleeComp.getLoc())
          .append("component declared here");
    }
  }

  std::vector<mlir::Value> preparedArguments;
  prepareArguments(adaptor.getArgs(), constructorTypes, op->getLoc(), rewriter, preparedArguments);

  auto funcPtr = rewriter.create<Zmir::ConstructorRefOp>(op.getLoc(), calleeComp);
  rewriter.replaceOpWithNewOp<mlir::func::CallIndirectOp>(op, funcPtr, preparedArguments);
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
  }

  assert(preparedArgs.size() == constructorTypes.size() && "incorrect number of arguments");
}

mlir::Value ZhlConstructLowering::prepareArgument(
    mlir::Value arg, mlir::Type expectedType, mlir::Location loc,
    mlir::ConversionPatternRewriter &rewriter
) const {
  return getTypeConverter()->materializeTargetConversion(rewriter, loc, expectedType, arg);
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
  auto lhs = getTypeConverter()->materializeTargetConversion(
      rewriter, op->getLoc(), Zmir::ValType::get(getContext()), {adaptor.getLhs()}
  );
  auto rhs = getTypeConverter()->materializeTargetConversion(
      rewriter, op->getLoc(), Zmir::ValType::get(getContext()), {adaptor.getRhs()}
  );
  rewriter.replaceOpWithNewOp<Zmir::ConstrainOp>(op, lhs, rhs);
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlGlobalRemoval
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlGlobalRemoval::matchAndRewrite(
    zirgen::Zhl::GlobalOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  rewriter.replaceOpWithNewOp<Zmir::NopOp>(
      op, mlir::TypeRange({Zmir::ComponentType::get(rewriter.getContext(), "Component")}),
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
      op, mlir::TypeRange({Zmir::ComponentType::get(rewriter.getContext(), "Component")}),
      mlir::ValueRange()
  );
  return mlir::success();
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

  valueOp->setAttr("writes_into", mlir::StringAttr::get(op.getContext(), op.getFieldName()));
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
  rewriter.create<Zmir::FieldDefOp>(loc, name, type);
}

mlir::LogicalResult ZhlDefineLowering::matchAndRewrite(
    zirgen::Zhl::DefinitionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  Zhl::DeclarationOp declr = op.getDeclaration().getDefiningOp<Zhl::DeclarationOp>();
  if (!declr) {
    return op.emitError("definition does not depend on a declaration");
  }
  auto name = declr.getMemberAttr();

  auto value = findTypeInUseDefChain(adaptor.getDefinition(), getTypeConverter());

  auto comp = op->getParentOfType<Zmir::ComponentOp>();
  assert(comp);
  createField(comp, name, value.getType(), rewriter, declr.getLoc());

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
mlir::FailureOr<mlir::Type> deduceTypeOfKnownExtern(mlir::StringRef name, mlir::MLIRContext *ctx) {
  if (name == "Log") {
    return Zmir::ComponentType::get(ctx, "Component");
  }

  return mlir::failure();
}

mlir::LogicalResult ZhlSuperLoweringInFunc::matchAndRewrite(
    zirgen::Zhl::SuperOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  if (!mlir::isa<mlir::func::FuncOp>(op->getParentOp())) {
    return op.emitError("lowering of super ops inside blocks is not defined yet");
  }
  auto comp = op->getParentOfType<Zmir::ComponentInterface>();
  assert(comp);
  auto self = rewriter.create<Zmir::GetSelfOp>(op.getLoc(), comp.getType());

  auto deducedType = deduceTypeOfKnownExtern(comp.getName(), rewriter.getContext());

  mlir::Type type;
  mlir::Value value;
  if (mlir::succeeded(deducedType)) {
    type = *deducedType;
    value = getTypeConverter()->materializeTargetConversion(
        rewriter, op->getLoc(), type, adaptor.getValue()
    );
    auto v = findTypeInUseDefChain(value, getTypeConverter());
    if (v.getType() == type) {
      value = v;
    }
  } else {
    value = findTypeInUseDefChain(adaptor.getValue(), getTypeConverter());
    type = value.getType();
  }

  createField(comp, "$super", type, rewriter, op.getLoc());
  auto writeOp = rewriter.replaceOpWithNewOp<Zmir::WriteFieldOp>(op, self, "$super", value);
  maybeAnnotateConstructorCallWithField(writeOp, value);

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

    return typeNameOp.emitError() << "unrecognized type " << name;
  }
}

inline mlir::FailureOr<mlir::Type>
getExternRetType(Zhl::ExternOp op, mlir::ConversionPatternRewriter &rewriter) {

  auto deducedType = deduceTypeOfKnownExtern(op.getName(), rewriter.getContext());
  if (mlir::succeeded(deducedType)) {
    return deducedType;
  }

  auto expr = mlir::dyn_cast<Zhl::GlobalOp>(op.getReturnType().getDefiningOp());
  if (!expr) {
    return mlir::failure();
  }

  return getTypeFromName(expr);
}

mlir::LogicalResult ZhlExternLowering::matchAndRewrite(
    Zhl::ExternOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {

  auto comp = op->getParentOfType<Zmir::ComponentOp>();
  if (!comp) {
    return mlir::failure();
  }

  std::vector<mlir::Value> args;
  std::transform(
      adaptor.getArgs().begin(), adaptor.getArgs().end(), std::back_inserter(args),
      [&](mlir::Value v) { return findTypeInUseDefChain(v, getTypeConverter()); }
  );

  std::vector<mlir::Type> argTypes;
  for (auto arg : args) {
    argTypes.push_back(arg.getType());
  }
  auto retType = getExternRetType(op, rewriter);
  if (mlir::failed(retType)) {
    return mlir::failure();
  }
  auto funcType = rewriter.getFunctionType(argTypes, {*retType});
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
  auto comp = findTypeInUseDefChain(adaptor.getComponent(), getTypeConverter());
  auto compType = mlir::dyn_cast<Zmir::ComponentType>(comp.getType());
  assert(compType && "expected a component type");
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  assert(mod && "expected a module as one of the parents of the op");
  mlir::SymbolTableCollection st;
  auto compOp = compType.getDefinition(st, mod);
  auto t =
      compOp.lookupFieldType(mlir::SymbolRefAttr::get(rewriter.getStringAttr(adaptor.getMember())));
  if (mlir::failed(t)) {
    return op->emitError() << "field '" << adaptor.getMember() << "' not found for component '"
                           << compOp.getName() << "'";
  }

  rewriter.replaceOpWithNewOp<Zmir::ReadFieldOp>(op, *t, comp, adaptor.getMember());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlSubscriptLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlSubscriptLowering::matchAndRewrite(
    Zhl::SubscriptOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  auto arr = findTypeInUseDefChain(adaptor.getArray(), getTypeConverter());
  auto elem = findTypeInUseDefChain(adaptor.getElement(), getTypeConverter());
  if (auto arrType = mlir::dyn_cast<Zmir::ArrayType>(arr.getType())) {
    rewriter.replaceOpWithNewOp<Zmir::ReadArrayOp>(op, arrType.getInnerType(), arr, elem);
  } else {
    rewriter.replaceOpWithNewOp<Zmir::ReadArrayOp>(op, arr.getType(), arr, elem);
  }
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlArrayLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlArrayLowering::matchAndRewrite(
    Zhl::ArrayOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  if (adaptor.getElements().empty()) {
    rewriter.replaceOpWithNewOp<Zmir::AllocArrayOp>(
        op, Zmir::UnboundedArrayType::get(
                rewriter.getContext(), Zmir::PendingType::get(rewriter.getContext())
            )
    );
    return mlir::success();
  }

  llvm::SmallVector<mlir::Value> args;
  findTypesInUseDefChain(adaptor.getElements(), getTypeConverter(), args);
  rewriter.replaceOpWithNewOp<Zmir::NewArrayOp>(
      op,
      Zmir::BoundedArrayType::get(
          rewriter.getContext(), args.front().getType(),
          rewriter.getIndexAttr(adaptor.getElements().size())
      ),
      args
  );
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlCompToZmirCompPattern
///////////////////////////////////////////////////////////

class ZhlCompToZmirCompPatternImpl {
public:
  using OpAdaptor = zirgen::Zhl::ComponentOp::Adaptor;
  ZhlCompToZmirCompPatternImpl(
      zirgen::Zhl::ComponentOp prev, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
  )
      : prev(prev), adaptor(adaptor), rewriter(rewriter) {}

  mlir::LogicalResult matchAndRewrite() {
    auto newCompOp = createComponent();
    mlir::OpBuilder::InsertionGuard compInsertionGuard(rewriter);
    auto *block = rewriter.createBlock(&newCompOp.getRegion());
    rewriter.setInsertionPointToStart(block);

    auto types = deduceConstructorArgTypes();
    if (mlir::failed(types)) {
      return mlir::failure();
    }

    auto arity = getComponentConstructorArity(prev);
    if (types->size() != arity.paramCount) {
      return prev->emitError() << "Inconsistent parameter detection result. Type detection "
                               << types->size() << ", arity detection " << arity.paramCount;
    }

    createComponentBody(newCompOp, arity, *types);
    rewriter.replaceOp(prev.getOperation(), newCompOp.getOperation());
    return mlir::success();
  }

private:
  mlir::FailureOr<std::map<uint32_t, mlir::Type>> deduceConstructorArgTypes() {
    std::map<uint32_t, mlir::Type> types;
    for (auto param : prev.getOps<Zhl::ConstructorParamOp>()) {

      auto type = inferType(param.getType(), param.getOperation());
      if (mlir::failed(type)) {
        return mlir::failure();
      }
      types.insert({param.getIndex(), *type});
    }

    return types;
  }

  mlir::FailureOr<mlir::Type>
  inferType(mlir::TypedValue<Zhl::ExprType> paramType, mlir::Operation *op) {
    // Option 1: The type is declared with zhl.global
    Zhl::GlobalOp typeNameOp = paramType.getDefiningOp<Zhl::GlobalOp>();
    if (typeNameOp) {
      auto type = getTypeFromName(typeNameOp);
      if (mlir::failed(type)) {
        return mlir::failure();
      } else {
        return type;
      }
    }
    // Option 2: The type is declared with zhl.specialize
    Zhl::SpecializeOp specializeOp = paramType.getDefiningOp<Zhl::SpecializeOp>();
    if (specializeOp) {
      auto t = inferType(specializeOp.getType(), specializeOp.getOperation());
      if (mlir::failed(t)) {
        return mlir::failure();
      }

      // What to do with the arguments of the specialization is TBD
      return t;
    }

    // Fail if none of the options match
    return op->emitError("could not detect the type of the parameter");
  }

  mlir::MLIRContext *getContext() { return rewriter.getContext(); }

  /*mlir::FailureOr<mlir::Type> getTypeFromName(Zhl::GlobalOp typeNameOp) {*/
  /*  auto name = typeNameOp.getName();*/
  /*  // Simple algorithm for now*/
  /*  if (name == "Val") {*/
  /*    return Zmir::ValType::get(getContext());*/
  /*  } else if (name == "String") {*/
  /*    return Zmir::StringType::get(getContext());*/
  /*  } else {*/
  /*    auto sym = mlir::SymbolTable::lookupNearestSymbolFrom(*/
  /*        typeNameOp.getOperation(), rewriter.getStringAttr(name));*/
  /*    if (sym != nullptr) {*/
  /*      auto op = mlir::cast<Zmir::ComponentOp>(sym);*/
  /*      if (!op) {*/
  /*        return typeNameOp.emitError()*/
  /*               << "type " << name << " is not a component";*/
  /*      }*/
  /*      return op.getType();*/
  /*    }*/
  /**/
  /*    return typeNameOp.emitError() << "unrecognized type " << name;*/
  /*  }*/
  /*}*/

  zkc::Zmir::ComponentOp createComponent() {
    auto maybeBuiltin = mlir::SymbolTable::lookupSymbolIn(
        prev->getParentOfType<mlir::ModuleOp>().getOperation(), prev.getName()
    );
    if (maybeBuiltin) {
      rewriter.eraseOp(maybeBuiltin);
    }

    auto attrs = prev->getAttrs();

    if (prev.getGeneric()) {
      std::vector<mlir::StringRef> typeParams;
      std::vector<mlir::StringRef> constParams;
      return rewriter.create<zkc::Zmir::ComponentOp>(
          prev.getLoc(), prev.getName(), typeParams, constParams, attrs
      );
    } else {
      return rewriter.create<zkc::Zmir::ComponentOp>(prev.getLoc(), prev.getName(), attrs);
    }
  }

  std::vector<mlir::NamedAttribute> funcBodyAttrs() {
    return {mlir::NamedAttribute(
        rewriter.getStringAttr("sym_visibility"), rewriter.getStringAttr("nested")
    )};
  }
  std::vector<mlir::Type>
  getArgTypes(ComponentArity &arity, std::map<uint32_t, mlir::Type> &types) {
    std::vector<mlir::Type> argTypes;

    for (uint32_t i = 0; i < arity.paramCount; i++) {
      // If the component is variadic mark the last argument as such
      if (i == arity.paramCount - 1 && arity.isVariadic) {
        argTypes.push_back(Zmir::VarArgsType::get(getContext(), types.at(i)));
        continue;
      }
      argTypes.push_back(types.at(i));
    }

    return argTypes;
  }

  void createComponentBody(
      zkc::Zmir::ComponentOp newOp, ComponentArity &arity, std::map<uint32_t, mlir::Type> types
  ) {
    auto argTypes = getArgTypes(arity, types);
    mlir::Type compType = newOp.getType();
    auto funcType = rewriter.getFunctionType(argTypes, {compType});
    auto attrs = funcBodyAttrs();

    auto bodyOp = rewriter.create<mlir::func::FuncOp>(
        prev.getLoc(), newOp.getBodyFuncName(), funcType, attrs
    );
    bodyOp.getRegion().takeBody(prev.getRegion());

    // Create arguments for the entry block (aka region arguments)
    auto &entryBlock = bodyOp.front();
    entryBlock.addArguments(argTypes, arity.locs);

    // Fill out epilogue
    auto *epilogue = bodyOp.addBlock();
    fillEpilogue(epilogue, newOp, rewriter);
  }

  void fillEpilogue(
      mlir::Block *block, zkc::Zmir::ComponentOp newOp, mlir::ConversionPatternRewriter &rewriter
  ) {
    mlir::OpBuilder::InsertionGuard insertionGuard(rewriter);
    rewriter.setInsertionPointToEnd(block);

    mlir::Location unk = rewriter.getUnknownLoc();
    // Prologue of Zmir components
    auto self = rewriter.create<zkc::Zmir::GetSelfOp>(unk, newOp.getType());
    rewriter.create<mlir::func::ReturnOp>(unk, mlir::ValueRange({self}));
  }

  zirgen::Zhl::ComponentOp prev;
  OpAdaptor adaptor;
  mlir::ConversionPatternRewriter &rewriter;
};

mlir::LogicalResult ZhlCompToZmirCompPattern::matchAndRewrite(
    zirgen::Zhl::ComponentOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  ZhlCompToZmirCompPatternImpl impl(op, adaptor, rewriter);
  return impl.matchAndRewrite();
}

///////////////////////////////////////////////////////////
/// ZhlRangeOpLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlRangeOpLowering::matchAndRewrite(
    Zhl::RangeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter
) const {
  // Allocate an unbounded array of vals
  auto arrAlloc = rewriter.replaceOpWithNewOp<Zmir::AllocArrayOp>(
      op, Zmir::UnboundedArrayType::get(
              rewriter.getContext(), Zmir::ValType::get(rewriter.getContext())
          )
  );
  // Create a for loop op using the operands as bounds
  auto one = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), 1);
  auto start = rewriter.create<Zmir::ValToIndexOp>(
      op.getStart().getLoc(), findTypeInUseDefChain(adaptor.getStart(), getTypeConverter())
  );
  auto end = rewriter.create<Zmir::ValToIndexOp>(
      op.getEnd().getLoc(), findTypeInUseDefChain(adaptor.getEnd(), getTypeConverter())
  );
  rewriter.create<mlir::scf::ForOp>(
      op.getLoc(), start, end, one, mlir::ValueRange(),
      [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange) {
    auto conv = builder.create<Zmir::IndexToValOp>(loc, iv);
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
  auto arrValue = findTypeInUseDefChain(adaptor.getArray(), getTypeConverter());
  if (!(mlir::isa<Zmir::ArrayType>(arrValue.getType()) ||
        mlir::isa<Zmir::PendingType>(arrValue.getType()))) {
    return op.emitOpError()
        .append("was expecting an array type but got ", arrValue.getType())
        .attachNote(arrValue.getLoc())
        .append("defined here");
  }

  int64_t size = -1;
  if (auto a = mlir::dyn_cast<Zmir::BoundedArrayType>(arrValue.getType())) {
    size = a.getSizeInt();
  }
  auto inner = tryGetArrayInnerType(arrValue.getType());
  auto finalType = resultArrayType(inner, size, getContext(), rewriter);

  auto arrAlloc = rewriter.create<Zmir::AllocArrayOp>(op.getLoc(), finalType);
  auto one = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), 1);
  auto zero = rewriter.create<mlir::index::ConstantOp>(op.getLoc(), 0);
  auto len = (size >= 0) ? rewriter.create<mlir::index::ConstantOp>(op.getLoc(), size)
                         : rewriter.create<Zmir::GetArrayLenOp>(op.getLoc(), arrValue);

  auto loop = rewriter.create<mlir::scf::ForOp>(
      op.getLoc(), zero, len->getResult(0), one, mlir::ValueRange(arrAlloc),
      [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange args) {
    auto itVal = builder.create<Zmir::ReadArrayOp>(loc, inner, arrValue, mlir::ValueRange(iv));
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
  mlir::SmallVector<mlir::Type> types;
  auto convRes = getTypeConverter()->convertTypes(op->getResultTypes(), types);
  if (mlir::failed(convRes)) {
    return op->emitError("failed to convert types from zhl to zmir");
  }
  auto exec = rewriter.replaceOpWithNewOp<mlir::scf::ExecuteRegionOp>(op, types);
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

  auto regionOp = mlir::cast<mlir::scf::ExecuteRegionOp>(parent);
  auto value = findTypeInUseDefChain(adaptor.getValue(), getTypeConverter());
  auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
      op.getLoc(), regionOp.getResultTypes(), mlir::ValueRange(value)
  );

  rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, cast.getResults());
  return mlir::success();
}

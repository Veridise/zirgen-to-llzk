#include "Patterns.h"
#include "ZMIRTypeConverter.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include "ZirToZkir/Passes/ConvertZhlToZmir/Helpers.h"
#include <cstdint>
#include <cstdio>
#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace zirgen;
using namespace zkc;

///////////////////////////////////////////////////////////
/// ZhlLiteralLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlLiteralLowering::matchAndRewrite(
    Zhl::LiteralOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  Zmir::ValType val = Zmir::ValType::get(getContext());
  rewriter.replaceOpWithNewOp<Zmir::LitValOp>(op, val, adaptor.getValue());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlLiteralStrLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlLiteralStrLowering::matchAndRewrite(
    Zhl::StringOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto val = Zmir::StringType::get(getContext());
  rewriter.replaceOpWithNewOp<Zmir::LitStrOp>(op, val, adaptor.getValue());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlParameterLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlParameterLowering::matchAndRewrite(
    Zhl::ConstructorParamOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto body = op->getParentOfType<mlir::func::FuncOp>();
  mlir::BlockArgument arg = body.getArgument(adaptor.getIndex());

  rewriter.replaceAllUsesWith(op, arg);
  rewriter.eraseOp(op);
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlConstructLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlConstructLowering::matchAndRewrite(
    Zhl::ConstructOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  Zhl::GlobalOp typeNameOp = op.getType().getDefiningOp<Zhl::GlobalOp>();
  if (!typeNameOp) {
    return op.emitError("constructed type is not declared by a global");
  }

  auto calleeComp =
      op->getParentOfType<mlir::ModuleOp>().lookupSymbol<Zmir::ComponentOp>(
          typeNameOp.getNameAttr());
  if (!calleeComp)
    return op->emitError() << "could not find component with name "
                           << typeNameOp.getNameAttr();

  auto funcPtr =
      rewriter.create<Zmir::ConstructorRefOp>(op.getLoc(), calleeComp);
  auto constructorTypes =
      calleeComp.getBodyFunc().getFunctionType().getInputs();
  if (!constructorTypes.empty()) {
    if (mlir::isa<Zmir::VarArgsType>(constructorTypes.back())) {
      auto rem = adaptor.getArgs().size() - constructorTypes.size();
      auto splitPoint = adaptor.getArgs().begin() + rem;

      std::vector<mlir::Value> normalArgs(adaptor.getArgs().begin(),
                                          splitPoint);
      llvm::dbgs() << "args = " << adaptor.getArgs().size() << "\n";
      llvm::dbgs() << "types = " << constructorTypes.size() << "\n";
      llvm::dbgs() << "normalArgs = " << normalArgs.size() << "\n";
      mlir::ValueRange varArgs(
          llvm::iterator_range(splitPoint, adaptor.getArgs().end()));
      llvm::dbgs() << "varArgs = " << varArgs.size() << "\n";
      auto va = rewriter.create<Zmir::VarArgsOp>(
          op.getLoc(), constructorTypes.back(), varArgs);
      llvm::dbgs() << "va = " << va << "\n";
      normalArgs.push_back(va);
      llvm::dbgs() << "normalArgs (after) = " << normalArgs.size() << "\n";

      rewriter.replaceOpWithNewOp<mlir::func::CallIndirectOp>(op, funcPtr,
                                                              normalArgs);

    } else {
      rewriter.replaceOpWithNewOp<mlir::func::CallIndirectOp>(
          op, funcPtr, adaptor.getArgs());
    }
  } else {
    rewriter.replaceOpWithNewOp<mlir::func::CallIndirectOp>(op, funcPtr,
                                                            mlir::ValueRange());
  }
  return mlir::success();
}

mlir::FailureOr<mlir::Type>
ZhlConstructLowering::getTypeFromName(mlir::StringRef name) const {
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
    Zhl::ConstraintOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<Zmir::ConstrainOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlGlobalRemoval
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlGlobalRemoval::matchAndRewrite(
    zirgen::Zhl::GlobalOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<Zmir::NopOp>(
      op,
      mlir::TypeRange(
          {Zmir::ComponentType::get(rewriter.getContext(), "Component")}),
      mlir::ValueRange());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlDeclarationRemoval
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlDeclarationRemoval::matchAndRewrite(
    zirgen::Zhl::DeclarationOp op, OpAdaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<Zmir::NopOp>(
      op,
      mlir::TypeRange(
          {Zmir::ComponentType::get(rewriter.getContext(), "Component")}),
      mlir::ValueRange());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlDefinitionLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult createField(Zmir::ComponentOp comp, mlir::StringRef name,
                                mlir::Type type,
                                mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location loc) {
  if (!comp)
    return mlir::failure();

  mlir::OpBuilder::InsertionGuard guard(rewriter);

  rewriter.setInsertionPointToStart(&comp.getRegion().front());
  rewriter.create<Zmir::FieldDefOp>(loc, name, type);
  return mlir::success();
}

mlir::LogicalResult ZhlDefineLowering::matchAndRewrite(
    zirgen::Zhl::DefinitionOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  Zhl::DeclarationOp declr =
      op.getDeclaration().getDefiningOp<Zhl::DeclarationOp>();
  if (!declr)
    return op.emitError("definition does not depend on a declaration");
  auto name = declr.getMemberAttr();
  // XXX: Do I need to put a pending type here?
  auto type = adaptor.getDefinition().getType();

  auto comp = op->getParentOfType<Zmir::ComponentOp>();
  if (!comp)
    return mlir::failure();
  auto fieldResult = createField(comp, name, type, rewriter, declr.getLoc());
  if (mlir::failed(fieldResult))
    return mlir::failure();

  auto self = rewriter.create<Zmir::GetSelfOp>(op.getLoc(), comp.getType());
  rewriter.replaceOpWithNewOp<Zmir::WriteFieldOp>(op, self, name,
                                                  adaptor.getDefinition());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlSuperLowering
///////////////////////////////////////////////////////////

/// A best-effort check of known externs to the type
/// they return.
mlir::FailureOr<mlir::Type> deduceTypeOfKnownExtern(mlir::StringRef name,
                                                    mlir::MLIRContext *ctx) {
  if (name == "Log") {
    return Zmir::ComponentType::get(ctx, "Component");
  }

  return mlir::failure();
}

mlir::LogicalResult ZhlSuperLowering::matchAndRewrite(
    zirgen::Zhl::SuperOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (!mlir::isa<mlir::func::FuncOp>(op->getParentOp()))
    return op.emitError(
        "lowering of super ops inside blocks is not defined yet");
  auto comp = op->getParentOfType<Zmir::ComponentOp>();
  if (!comp)
    return mlir::failure();
  auto deducedType =
      deduceTypeOfKnownExtern(comp.getName(), rewriter.getContext());
  auto type = mlir::succeeded(deducedType)
                  ? *deducedType
                  : Zmir::PendingType::get(rewriter.getContext());

  llvm::dbgs() << "Type for $super type is " << type << " from "
               << adaptor.getValue() << "\n";
  auto fieldResult = createField(comp, "$super", type, rewriter, op.getLoc());
  if (mlir::failed(fieldResult))
    return mlir::failure();
  auto self = rewriter.create<Zmir::GetSelfOp>(op.getLoc(), comp.getType());
  rewriter.replaceOpWithNewOp<Zmir::WriteFieldOp>(op, self, "$super",
                                                  adaptor.getValue());

  // Find the prologue and join it to the current block
  auto *prologue = &comp.getBodyFunc().getRegion().back();
  auto nop = rewriter.create<Zmir::NopOp>(
      rewriter.getUnknownLoc(), mlir::TypeRange(), mlir::ValueRange());
  rewriter.inlineBlockBefore(prologue, nop.getOperation());
  rewriter.eraseOp(nop);
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlExternLowering
///////////////////////////////////////////////////////////

std::vector<mlir::NamedAttribute>
externFuncAttrs(mlir::ConversionPatternRewriter &rewriter) {
  return {mlir::NamedAttribute(rewriter.getStringAttr("extern"),
                               rewriter.getUnitAttr()),
          mlir::NamedAttribute(rewriter.getStringAttr("sym_visibility"),
                               rewriter.getStringAttr("private"))};
}

mlir::FailureOr<mlir::func::FuncOp> createExternFunc(
    Zmir::ComponentOp op, llvm::StringRef name, mlir::FunctionType type,
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op.getBodyFunc());
  auto attrs = externFuncAttrs(rewriter);

  return rewriter.create<mlir::func::FuncOp>(loc, name, type, attrs);
}

mlir::LogicalResult ZhlExternLowering::matchAndRewrite(
    Zhl::ExternOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  auto comp = op->getParentOfType<Zmir::ComponentOp>();
  if (!comp)
    return mlir::failure();
  std::vector<mlir::Type> argTypes;
  for (auto arg : adaptor.getArgs()) {
    argTypes.push_back(arg.getType());
  }

  auto deducedType =
      deduceTypeOfKnownExtern(op.getName(), rewriter.getContext());
  auto retType = mlir::succeeded(deducedType)
                     ? *deducedType
                     : Zmir::PendingType::get(rewriter.getContext());
  auto funcType = rewriter.getFunctionType(argTypes, {retType});
  std::string externName(op.getName().str());
  externName += "$$extern";
  auto externDeclrResult =
      createExternFunc(comp, externName, funcType, rewriter, op.getLoc());
  if (mlir::failed(externDeclrResult))
    return mlir::failure();

  rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
      op, *externDeclrResult, mlir::ValueRange(adaptor.getArgs()));

  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlLookupLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlLookupLowering::matchAndRewrite(
    Zhl::LookupOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<Zmir::ReadFieldOp>(
      op, adaptor.getComponent().getType(), adaptor.getComponent(),
      adaptor.getMember());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlSubscriptLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlSubscriptLowering::matchAndRewrite(
    Zhl::SubscriptOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<Zmir::ReadArrayOp>(
      op, adaptor.getArray().getType(), adaptor.getArray(),
      adaptor.getElement());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlArrayLowering
///////////////////////////////////////////////////////////

mlir::LogicalResult ZhlArrayLowering::matchAndRewrite(
    Zhl::ArrayOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (adaptor.getElements().empty()) {
    rewriter.replaceOpWithNewOp<Zmir::AllocArrayOp>(
        op, Zmir::ArrayType::get(rewriter.getContext(),
                                 Zmir::PendingType::get(rewriter.getContext()),
                                 rewriter.getIndexAttr(0)));
    return mlir::success();
  }
  rewriter.replaceOpWithNewOp<Zmir::NewArrayOp>(
      op,
      Zmir::ArrayType::get(rewriter.getContext(),
                           adaptor.getElements().getType().front(),
                           rewriter.getIndexAttr(adaptor.getElements().size())),
      adaptor.getElements());
  return mlir::success();
}

///////////////////////////////////////////////////////////
/// ZhlCompToZmirCompPattern
///////////////////////////////////////////////////////////

class ZhlCompToZmirCompPatternImpl {
public:
  using OpAdaptor = zirgen::Zhl::ComponentOp::Adaptor;
  ZhlCompToZmirCompPatternImpl(zirgen::Zhl::ComponentOp prev, OpAdaptor adaptor,
                               mlir::ConversionPatternRewriter &rewriter)
      : prev(prev), adaptor(adaptor), rewriter(rewriter) {}

  mlir::LogicalResult matchAndRewrite() {
    auto newCompOp = createComponent();
    mlir::OpBuilder::InsertionGuard compInsertionGuard(rewriter);
    auto *block = rewriter.createBlock(&newCompOp.getRegion());
    rewriter.setInsertionPointToStart(block);

    auto types = deduceConstructorArgTypes();
    if (mlir::failed(types))
      return mlir::failure();

    auto arity = getComponentConstructorArity(prev);
    if (types->size() != arity.paramCount)
      return prev->emitError()
             << "Inconsistent parameter detection result. Type detection "
             << types->size() << ", arity detection " << arity.paramCount;

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
    Zhl::SpecializeOp specializeOp =
        paramType.getDefiningOp<Zhl::SpecializeOp>();
    if (specializeOp) {
      auto t = inferType(specializeOp.getType(), specializeOp.getOperation());
      if (mlir::failed(t))
        return mlir::failure();

      // What to do with the arguments of the specialization is TBD
      return t;
    }

    // Fail if none of the options match
    return op->emitError("could not detect the type of the parameter");
  }

  mlir::MLIRContext *getContext() { return rewriter.getContext(); }

  mlir::FailureOr<mlir::Type> getTypeFromName(Zhl::GlobalOp typeNameOp) {
    auto name = typeNameOp.getName();
    // Simple algorithm for now
    if (name == "Val") {
      return Zmir::ValType::get(getContext());
    } else if (name == "String") {
      return Zmir::StringType::get(getContext());
    } else {
      auto sym = mlir::SymbolTable::lookupNearestSymbolFrom(
          typeNameOp.getOperation(), rewriter.getStringAttr(name));
      if (sym != nullptr) {
        auto op = mlir::cast<Zmir::ComponentOp>(sym);
        if (!op) {
          return typeNameOp.emitError()
                 << "type " << name << " is not a component";
        }
        return op.getType();
      }

      return typeNameOp.emitError() << "unrecognized type " << name;
    }
  }

  zkc::Zmir::ComponentOp createComponent() {
    auto maybeBuiltin = mlir::SymbolTable::lookupSymbolIn(
        prev->getParentOfType<mlir::ModuleOp>().getOperation(), prev.getName());
    if (maybeBuiltin)
      rewriter.eraseOp(maybeBuiltin);

    auto attrs = prev->getAttrs();

    if (prev.getGeneric()) {
      std::vector<mlir::StringRef> typeParams;
      std::vector<mlir::StringRef> constParams;
      return rewriter.create<zkc::Zmir::ComponentOp>(
          prev.getLoc(), prev.getName(), typeParams, constParams, attrs);
    } else {
      return rewriter.create<zkc::Zmir::ComponentOp>(prev.getLoc(),
                                                     prev.getName(), attrs);
    }
  }

  std::vector<mlir::NamedAttribute> funcBodyAttrs() {
    return {mlir::NamedAttribute(rewriter.getStringAttr("sym_visibility"),
                                 rewriter.getStringAttr("nested"))};
  }
  std::vector<mlir::Type> getArgTypes(ComponentArity &arity,
                                      std::map<uint32_t, mlir::Type> &types) {
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

  void createComponentBody(zkc::Zmir::ComponentOp newOp, ComponentArity &arity,
                           std::map<uint32_t, mlir::Type> types) {
    auto argTypes = getArgTypes(arity, types);
    mlir::Type compType = newOp.getType();
    auto funcType = rewriter.getFunctionType(argTypes, {compType});
    auto attrs = funcBodyAttrs();

    auto bodyOp = rewriter.create<mlir::func::FuncOp>(
        prev.getLoc(), newOp.getBodyFuncName(), funcType, attrs);
    bodyOp.getRegion().takeBody(prev.getRegion());

    // Create arguments for the entry block (aka region arguments)
    auto &entryBlock = bodyOp.front();
    entryBlock.addArguments(argTypes, arity.locs);

    // Fill out prologue
    auto *prologue = bodyOp.addBlock();
    fillPrologue(prologue, newOp, rewriter);
  }

  void fillPrologue(mlir::Block *block, zkc::Zmir::ComponentOp newOp,
                    mlir::ConversionPatternRewriter &rewriter) {
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
    zirgen::Zhl::ComponentOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  ZhlCompToZmirCompPatternImpl impl(op, adaptor, rewriter);
  return impl.matchAndRewrite();
}

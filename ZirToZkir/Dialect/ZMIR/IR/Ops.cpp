#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include <algorithm>
#include <iterator>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LogicalResult.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "ZirToZkir/Dialect/ZMIR/IR/Ops.inc.cpp"

namespace zkc::Zmir {

void ComponentOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        llvm::StringRef name, IsBuiltIn) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.getOrAddProperties<Properties>().builtin = builder.getUnitAttr();
  state.addRegion();
}

mlir::ArrayAttr fillParams(mlir::OpBuilder &builder,
                           mlir::OperationState &state,
                           mlir::ArrayRef<mlir::StringRef> params) {
  std::vector<mlir::Attribute> symbols;
  std::transform(params.begin(), params.end(), std::back_inserter(symbols),
                 [&](mlir::StringRef s) {
                   return mlir::SymbolRefAttr::get(
                       mlir::StringAttr::get(builder.getContext(), s));
                 });
  return builder.getArrayAttr(symbols);
}

void ComponentOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        llvm::StringRef name,
                        mlir::ArrayRef<mlir::StringRef> typeParams,
                        mlir::ArrayRef<mlir::StringRef> constParams,
                        IsBuiltIn) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.getOrAddProperties<Properties>().builtin = builder.getUnitAttr();
  if (typeParams.size() + constParams.size() > 1) {
    state.getOrAddProperties<Properties>().generic = builder.getUnitAttr();
    state.getOrAddProperties<Properties>().type_params =
        fillParams(builder, state, typeParams);
    state.getOrAddProperties<Properties>().const_params =
        fillParams(builder, state, constParams);
  }
  state.addRegion();
}

void ComponentOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        llvm::StringRef name,
                        mlir::ArrayRef<mlir::StringRef> typeParams,
                        mlir::ArrayRef<mlir::StringRef> constParams,
                        llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.getOrAddProperties<Properties>().type_params =
      fillParams(builder, state, typeParams);
  state.getOrAddProperties<Properties>().const_params =
      fillParams(builder, state, constParams);
  state.addAttributes(attrs);
  state.addRegion();
}

void ComponentOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        llvm::StringRef name,
                        llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.addAttributes(attrs);
  state.addRegion();
}

mlir::Type ComponentOp::getType() {
  llvm::dbgs() << "Type requested!!\n";
  if (getTypeParams().has_value() && getConstParams().has_value()) {
    auto typeParams = *getTypeParams();
    auto constParams = *getConstParams();
    return ComponentType::get(getContext(), getSymName(), getSuperType(),
                              typeParams.getValue(), constParams.getValue());
  } else {
    return ComponentType::get(getContext(), getSymName(), getSuperType());
  }
}

mlir::Type ComponentOp::getSuperType() {
  // Special case for the root component
  if (getSymName() == "Component")
    return ComponentType::get(getContext(), getSymName());

  mlir::SymbolTable st(this->getOperation());
  auto *op = st.lookup("$super");
  // If $super could not be found default to pending
  if (!op)
    return Zmir::PendingType::get(getContext());

  auto fieldDef = mlir::dyn_cast<Zmir::FieldDefOp>(op);
  assert(fieldDef &&
         "expecting a field definition op to be tied to the $super symbol");

  return fieldDef.getType();
}

mlir::SymbolRefAttr ComponentOp::getBodySym() {
  mlir::OpBuilder builder(getContext());
  return mlir::SymbolRefAttr::get(
      getSymNameAttr(),
      {mlir::SymbolRefAttr::get(builder.getStringAttr(getBodyFuncName()))});
}

::mlir::func::FuncOp ComponentOp::getBodyFunc() {
  mlir::OpBuilder builder(getContext());
  return mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
      getOperation(), builder.getStringAttr(getBodyFuncName()));
}

void ConstructorRefOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &state, ComponentOp op) {
  state.getOrAddProperties<Properties>().component =
      mlir::SymbolRefAttr::get(op.getSymNameAttr());
  state.addTypes({op.getBodyFunc().getFunctionType()});
}

mlir::LogicalResult ConstructorRefOp::verify() {
  mlir::StringRef compName = getComponent();
  mlir::Type type = getType();

  // Try to find the referenced component.
  auto comp =
      (*this)->getParentOfType<mlir::ModuleOp>().lookupSymbol<ComponentOp>(
          compName);
  if (!comp)
    return emitOpError() << "reference to undefined component '" << compName
                         << "'";

  // Check that the referenced component's constructor has the correct type.
  if (comp.getBodyFunc().getFunctionType() != type)
    return emitOpError("reference to constructor with mismatched type");

  return mlir::success();
}

bool ConstructorRefOp::isBuildableWith(mlir::Attribute value, mlir::Type type) {
  return llvm::isa<mlir::FlatSymbolRefAttr>(value) &&
         llvm::isa<mlir::FunctionType>(type);
}

mlir::LogicalResult
ReadFieldOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  // TODO
  return mlir::success();
}

mlir::LogicalResult
WriteFieldOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  // TODO
  return mlir::success();
}

mlir::LogicalResult GetGlobalOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location>,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::OpaqueProperties,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  // TODO
  inferredReturnTypes.push_back(ValType::get(ctx));
  return mlir::success();
}

} // namespace zkc::Zmir

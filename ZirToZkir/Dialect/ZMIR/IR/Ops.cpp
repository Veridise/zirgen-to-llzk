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
#include <string_view>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "ZirToZkir/Dialect/ZMIR/IR/OpInterfaces.inc.cpp"

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

void SplitComponentOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &state, llvm::StringRef name,
                             IsBuiltIn) {
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

void SplitComponentOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &state, llvm::StringRef name,
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

void SplitComponentOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &state, llvm::StringRef name,
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

void SplitComponentOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &state, llvm::StringRef name,
                             llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.addAttributes(attrs);
  state.addRegion();
}

mlir::Type ComponentOp::getType() {
  if (getTypeParams().has_value() && getConstParams().has_value()) {
    auto typeParams = *getTypeParams();
    auto constParams = *getConstParams();
    return ComponentType::get(getContext(), getSymName(), typeParams.getValue(),
                              constParams.getValue());
  } else {
    return ComponentType::get(getContext(), getSymName());
  }
}

mlir::Type SplitComponentOp::getType() {
  if (getTypeParams().has_value() && getConstParams().has_value()) {
    auto typeParams = *getTypeParams();
    auto constParams = *getConstParams();
    return ComponentType::get(getContext(), getSymName(), typeParams.getValue(),
                              constParams.getValue());
  } else {
    return ComponentType::get(getContext(), getSymName());
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

mlir::Type SplitComponentOp::getSuperType() {
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

#if 0
inline mlir::SymbolRefAttr makeCompFuncSymbol(mlir::MLIRContext *ctx,
                                              mlir::StringAttr parent,
                                              std::string_view name) {
  mlir::OpBuilder builder(ctx);
  return mlir::SymbolRefAttr::get(
      parent, {mlir::SymbolRefAttr::get(builder.getStringAttr(name))});
}

inline mlir::func::FuncOp getCompFunc(mlir::Operation *op,
                                      std::string_view name) {
  mlir::OpBuilder builder(op->getContext());
  return mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
      op, builder.getStringAttr(name));
}

// FIXME: It will be way more clean to have a common class for both types of
// components instead of this macro mess. I just can't remember how to do that
// with tablegen right now.
#define bodyFuncSym(T)                                                         \
  mlir::SymbolRefAttr T::getBodySym() {                                        \
    return makeCompFuncSymbol(getContext(), getSymNameAttr(),                  \
                              getBodyFuncName());                              \
  }
#define constrainFuncSym(T)                                                    \
  mlir::SymbolRefAttr T::getConstrainSym() {                                   \
    return makeCompFuncSymbol(getContext(), getSymNameAttr(),                  \
                              getConstrainFuncName());                         \
  }

#define bodyFunc(T)                                                            \
  mlir::func::FuncOp T::getBodyFunc() {                                        \
    return getCompFunc(getOperation(), getBodyFuncName());                     \
  }
#define constrainFunc(T)                                                       \
  mlir::func::FuncOp T::getConstrainFunc() {                                   \
    return getCompFunc(getOperation(), getConstrainFuncName());                \
  }

#define compFuncs(T)                                                           \
  bodyFuncSym(T) bodyFunc(T) constrainFuncSym(T) constrainFunc(T)
// clang-format off
compFuncs(ComponentOp) 
compFuncs(SplitComponentOp)
// clang-forman on
#undef bodyFuncSym
#undef constrainFuncSym
#undef bodyFunc
#undef constrainFuncSym
#undef compFuncs

#endif

void ConstructorRefOp::build(mlir::OpBuilder &builder,
                                 mlir::OperationState &state, DefinesBodyFunc op) {
  state.getOrAddProperties<Properties>().component =
      mlir::SymbolRefAttr::get(op.getNameAttr());
  state.addTypes({op.getBodyFunc().getFunctionType()});
}

mlir::LogicalResult ConstructorRefOp::verify() {
  mlir::StringRef compName = getComponent();
  mlir::Type type = getType();

  // Try to find the referenced component.
  auto comp =
      (*this)->getParentOfType<mlir::ModuleOp>().lookupSymbol<ComponentInterface>(
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

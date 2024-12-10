#include "ZirToZkir/Dialect/ZMIR/IR/Ops.h"
#include "ZirToZkir/Dialect/ZMIR/IR/Types.h"
#include "zkir/Dialect/ZKIR/IR/Ops.h"
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

void ComponentOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name, IsBuiltIn
) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.getOrAddProperties<Properties>().builtin = builder.getUnitAttr();
  state.addRegion();
}

void SplitComponentOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name, IsBuiltIn
) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.getOrAddProperties<Properties>().builtin = builder.getUnitAttr();
  state.addRegion();
}

mlir::ArrayAttr fillParams(
    mlir::OpBuilder &builder, mlir::OperationState &state, mlir::ArrayRef<mlir::StringRef> params
) {
  std::vector<mlir::Attribute> symbols;
  std::transform(params.begin(), params.end(), std::back_inserter(symbols), [&](mlir::StringRef s) {
    return mlir::SymbolRefAttr::get(mlir::StringAttr::get(builder.getContext(), s));
  });
  return builder.getArrayAttr(symbols);
}

void ComponentOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
    mlir::ArrayRef<mlir::StringRef> typeParams, mlir::ArrayRef<mlir::StringRef> constParams,
    IsBuiltIn
) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.getOrAddProperties<Properties>().builtin = builder.getUnitAttr();
  if (typeParams.size() + constParams.size() > 1) {
    state.getOrAddProperties<Properties>().generic = builder.getUnitAttr();
    state.getOrAddProperties<Properties>().type_params = fillParams(builder, state, typeParams);
    state.getOrAddProperties<Properties>().const_params = fillParams(builder, state, constParams);
  }
  state.addRegion();
}

void SplitComponentOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
    mlir::ArrayRef<mlir::StringRef> typeParams, mlir::ArrayRef<mlir::StringRef> constParams,
    IsBuiltIn
) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.getOrAddProperties<Properties>().builtin = builder.getUnitAttr();
  if (typeParams.size() + constParams.size() > 1) {
    state.getOrAddProperties<Properties>().generic = builder.getUnitAttr();
    state.getOrAddProperties<Properties>().type_params = fillParams(builder, state, typeParams);
    state.getOrAddProperties<Properties>().const_params = fillParams(builder, state, constParams);
  }
  state.addRegion();
}

void ComponentOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
    mlir::ArrayRef<mlir::StringRef> typeParams, mlir::ArrayRef<mlir::StringRef> constParams,
    llvm::ArrayRef<mlir::NamedAttribute> attrs
) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.getOrAddProperties<Properties>().type_params = fillParams(builder, state, typeParams);
  state.getOrAddProperties<Properties>().const_params = fillParams(builder, state, constParams);
  state.addAttributes(attrs);
  state.addRegion();
}

void SplitComponentOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
    mlir::ArrayRef<mlir::StringRef> typeParams, mlir::ArrayRef<mlir::StringRef> constParams,
    llvm::ArrayRef<mlir::NamedAttribute> attrs
) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.getOrAddProperties<Properties>().type_params = fillParams(builder, state, typeParams);
  state.getOrAddProperties<Properties>().const_params = fillParams(builder, state, constParams);
  state.addAttributes(attrs);
  state.addRegion();
}

void ComponentOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
    llvm::ArrayRef<mlir::NamedAttribute> attrs
) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.addAttributes(attrs);
  state.addRegion();
}

void SplitComponentOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
    llvm::ArrayRef<mlir::NamedAttribute> attrs
) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.addAttributes(attrs);
  state.addRegion();
}

mlir::Type ComponentOp::getType() {
  if (getTypeParams().has_value() && getConstParams().has_value()) {
    auto typeParams = *getTypeParams();
    auto constParams = *getConstParams();
    return ComponentType::get(
        getContext(), getSymName(), typeParams.getValue(), constParams.getValue()
    );
  } else {
    return ComponentType::get(getContext(), getSymName());
  }
}

mlir::Type SplitComponentOp::getType() {
  if (getTypeParams().has_value() && getConstParams().has_value()) {
    auto typeParams = *getTypeParams();
    auto constParams = *getConstParams();
    return ComponentType::get(
        getContext(), getSymName(), typeParams.getValue(), constParams.getValue()
    );
  } else {
    return ComponentType::get(getContext(), getSymName());
  }
}

template <typename CompOp> mlir::FailureOr<mlir::Type> getSuperTypeCommon(CompOp &op) {
  if (op.isRoot()) {
    if (auto comp = mlir::dyn_cast<zkc::Zmir::ComponentType>(op.getType())) {
      return comp;
    } else {
      return mlir::failure();
    }
  }
  return op.lookupFieldType(op.getSuperFieldName());
}

mlir::FailureOr<mlir::Type> ComponentOp::getSuperType() { return getSuperTypeCommon(*this); }

mlir::FailureOr<mlir::Type> SplitComponentOp::getSuperType() { return getSuperTypeCommon(*this); }

inline mlir::FailureOr<mlir::Type>
lookupFieldTypeCommon(mlir::FlatSymbolRefAttr fieldName, mlir::Operation *op) {

  auto fieldOp = mlir::SymbolTable::lookupNearestSymbolFrom<FieldDefOp>(op, fieldName);
  if (!fieldOp) {
    return mlir::failure();
  }

  return fieldOp.getType();
}

mlir::FailureOr<::mlir::Type> ComponentOp::lookupFieldType(mlir::FlatSymbolRefAttr fieldName) {
  return lookupFieldTypeCommon(fieldName, getOperation());
}

mlir::FailureOr<::mlir::Type> SplitComponentOp::lookupFieldType(mlir::FlatSymbolRefAttr fieldName) {
  return lookupFieldTypeCommon(fieldName, getOperation());
}

void ConstructorRefOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, ComponentInterface op
) {
  state.getOrAddProperties<Properties>().component = mlir::SymbolRefAttr::get(op.getNameAttr());
  if (op.getBuiltin()) {
    state.getOrAddProperties<Properties>().builtin = mlir::UnitAttr::get(builder.getContext());
  }
  state.addTypes({op.getBodyFunc().getFunctionType()});
}

mlir::LogicalResult ConstructorRefOp::verify() {
  mlir::StringRef compName = getComponent();
  mlir::Type type = getType();
  auto mod = (*this)->getParentOfType<mlir::ModuleOp>();

  // Try to find the referenced component.
  auto comp = mod.lookupSymbol<ComponentInterface>(compName);
  if (!comp) {
    // The constructor reference could be temporarly pointing
    // to a zkir struct. Assume it is correct if that's the case.
    auto structComp = mod.lookupSymbol<zkir::StructDefOp>(compName);
    if (structComp) {
      return mlir::success();
    }

    return emitOpError() << "reference to undefined component '" << compName << "'";
  }

  // Check that the referenced component's constructor has the correct type.
  if (comp.getBodyFunc().getFunctionType() != type) {
    return emitOpError("reference to constructor with mismatched type");
  }

  return mlir::success();
}

bool ConstructorRefOp::isBuildableWith(mlir::Attribute value, mlir::Type type) {
  return llvm::isa<mlir::FlatSymbolRefAttr>(value) && llvm::isa<mlir::FunctionType>(type);
}

mlir::LogicalResult ReadFieldOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  // TODO
  return mlir::success();
}

mlir::LogicalResult WriteFieldOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  // TODO
  return mlir::success();
}

mlir::LogicalResult GetGlobalOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location>, mlir::ValueRange operands,
    mlir::DictionaryAttr, mlir::OpaqueProperties, mlir::RegionRange,
    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes
) {
  // TODO
  inferredReturnTypes.push_back(ValType::get(ctx));
  return mlir::success();
}

} // namespace zkc::Zmir

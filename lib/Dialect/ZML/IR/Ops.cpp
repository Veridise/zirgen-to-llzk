#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <llvm/ADT/SmallVector.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/Types.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include <zklang/Dialect/ZML/IR/OpInterfaces.cpp.inc>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include <zklang/Dialect/ZML/IR/Ops.cpp.inc>

namespace zml {

void SelfOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type compType,
    mlir::function_ref<void(mlir::OpBuilder &, mlir::Value)> buildFn
) {
  state.addTypes(compType);
  auto *region = state.addRegion();
  region->emplaceBlock();
  assert(region->hasOneBlock());
  region->addArgument(compType, state.location);
  if (buildFn) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&region->front());
    buildFn(builder, region->getArgument(0));
  }
}

void SelfOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type compType,
    mlir::Region &movedRegion
) {
  state.addTypes(compType);
  auto *region = state.addRegion();
  region->takeBody(movedRegion);
  assert(region->hasOneBlock());
  region->addArgument(compType, state.location);
}

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
    mlir::ArrayRef<mlir::StringRef> params, IsBuiltIn
) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.getOrAddProperties<Properties>().builtin = builder.getUnitAttr();
  if (params.size() > 1) {
    state.getOrAddProperties<Properties>().generic = builder.getUnitAttr();
    state.getOrAddProperties<Properties>().params = fillParams(builder, state, params);
  }
  state.addRegion();
}

void SplitComponentOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
    mlir::ArrayRef<mlir::StringRef> params, IsBuiltIn
) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.getOrAddProperties<Properties>().builtin = builder.getUnitAttr();
  if (params.size() > 1) {
    state.getOrAddProperties<Properties>().generic = builder.getUnitAttr();
    state.getOrAddProperties<Properties>().params = fillParams(builder, state, params);
  }
  state.addRegion();
}

void ComponentOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
    mlir::ArrayRef<mlir::StringRef> typeParams, llvm::ArrayRef<mlir::NamedAttribute> attrs
) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.getOrAddProperties<Properties>().params = fillParams(builder, state, typeParams);
  state.addAttributes(attrs);
  state.addRegion();
}

void SplitComponentOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
    mlir::ArrayRef<mlir::StringRef> params, llvm::ArrayRef<mlir::NamedAttribute> attrs
) {
  state.getOrAddProperties<Properties>().sym_name = builder.getStringAttr(name);
  state.getOrAddProperties<Properties>().params = fillParams(builder, state, params);
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

template <typename CompOp> mlir::Type getSuperType(CompOp &op) {
  if (op.isRoot()) {
    return nullptr;
  }
  auto result = op.lookupFieldType(op.getSuperFieldName());
  assert(mlir::succeeded(result));
  return *result;
}

template <typename CompOp> mlir::Type getType(CompOp &op) {
  if (op.isRoot()) {
    return ComponentType::Component(op.getContext());
  }
  if (op.getParams().has_value()) {
    auto params = *op.getParams();
    return ComponentType::get(
        op.getContext(), op.getSymName(), getSuperType(op), params.getValue(), op.getBuiltin()
    );
  } else {
    return ComponentType::get(op.getContext(), op.getSymName(), getSuperType(op), op.getBuiltin());
  }
}

mlir::Type ComponentOp::getType() { return ::zml::getType(*this); }
mlir::Type SplitComponentOp::getType() { return ::zml::getType(*this); }

template <typename CompOp> mlir::FailureOr<mlir::Type> getSuperTypeCommon(CompOp &op) {
  if (op.isRoot()) {
    if (auto comp = mlir::dyn_cast<zml::ComponentType>(op.getType())) {
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
    mlir::OpBuilder &builder, mlir::OperationState &state, ComponentInterface op,
    mlir::FunctionType fnType
) {
  build(builder, state, mlir::SymbolRefAttr::get(op.getNameAttr()), fnType, op.getBuiltin());
}

void ConstructorRefOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, mlir::FlatSymbolRefAttr sym,
    mlir::FunctionType fnType, bool isBuiltin
) {
  state.getOrAddProperties<Properties>().component = sym;
  if (isBuiltin) {
    state.getOrAddProperties<Properties>().builtin = mlir::UnitAttr::get(builder.getContext());
  }
  state.addTypes({fnType});
}

mlir::LogicalResult checkConstructorTypeIsValid(
    mlir::FunctionType expected, mlir::Type provided,
    std::function<mlir::InFlightDiagnostic()> emitError
) {
  if (!mlir::isa<mlir::FunctionType>(provided)) {
    return emitError() << "has malformed IR: was expecting a function type";
  }
  auto providedFn = mlir::cast<mlir::FunctionType>(provided);

  if (expected.getInputs().size() != providedFn.getInputs().size()) {
    return emitError() << "was expecting " << expected.getInputs().size() << " arguments and got "
                       << providedFn.getInputs().size();
  }
  if (providedFn.getResults().size() != 1) {
    return emitError() << "was expecting 1 result and got " << providedFn.getResults().size();
  }

  // TODO: Need to check each individual type
  return mlir::success();
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
    auto structComp = mod.lookupSymbol<llzk::StructDefOp>(compName);
    if (structComp) {
      return mlir::success();
    }

    return emitOpError() << "reference to undefined component '" << compName << "'";
  }

  return checkConstructorTypeIsValid(comp.getBodyFunc().getFunctionType(), type, [&]() {
    return emitOpError();
  });
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
  inferredReturnTypes.push_back(ComponentType::Val(ctx));
  return mlir::success();
}

mlir::OpFoldResult LitValOp::fold(LitValOp::FoldAdaptor) { return getValueAttr(); }

mlir::OpFoldResult LitValArrayOp::fold(LitValArrayOp::FoldAdaptor) { return getElementsAttr(); }

mlir::OpFoldResult NewArrayOp::fold(NewArrayOp::FoldAdaptor adaptor) {
  mlir::SmallVector<long> values;
  for (auto attr : adaptor.getElements()) {
    if (!attr) {
      return nullptr;
    }
    if (auto arrayAttr = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
      values.insert(values.end(), arrayAttr.asArrayRef().begin(), arrayAttr.asArrayRef().end());
    } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
      values.push_back(intAttr.getInt());
    } else {
      return nullptr;
    }
  }

  return mlir::DenseI64ArrayAttr::get(getContext(), values);
}

} // namespace zml

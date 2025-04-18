//===- Ops.cpp - ZML operations ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

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
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/IR/Types.h>

using namespace llzk;

// TableGen'd implementation files
#define GET_OP_CLASSES
#include <zklang/Dialect/ZML/IR/OpInterfaces.cpp.inc>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include <zklang/Dialect/ZML/IR/Ops.cpp.inc>

using namespace mlir;

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
    mlir::OpBuilder &, mlir::OperationState &state, mlir::Type compType, mlir::Region &movedRegion
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
    mlir::OpBuilder &builder, mlir::OperationState &, mlir::ArrayRef<mlir::StringRef> params
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
  build(builder, state, mlir::SymbolRefAttr::get(op.getNameAttr()), 0, fnType, op.getBuiltin());
}

void ConstructorRefOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, ComponentInterface op,
    uint64_t liftedParams, mlir::FunctionType fnType
) {
  build(
      builder, state, mlir::SymbolRefAttr::get(op.getNameAttr()), liftedParams, fnType,
      op.getBuiltin()
  );
}

void ConstructorRefOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, mlir::FlatSymbolRefAttr sym,
    uint64_t liftedParams, mlir::FunctionType fnType, bool isBuiltin
) {
  Properties &properties = state.getOrAddProperties<Properties>();

  assert(liftedParams <= std::numeric_limits<int64_t>::max());
  properties.numLiftedParams = builder.getIndexAttr(static_cast<int64_t>(liftedParams));
  properties.component = sym;
  if (isBuiltin) {
    properties.builtin = mlir::UnitAttr::get(builder.getContext());
  }

  state.addTypes({fnType});
}

void ConstructorRefOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, mlir::FlatSymbolRefAttr sym,
    mlir::FunctionType fnType, bool isBuiltin
) {
  build(builder, state, sym, 0, fnType, isBuiltin);
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
    // to an LLZK struct. Assume it is correct if that's the case.
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

mlir::LogicalResult ReadFieldOp::verifySymbolUses(mlir::SymbolTableCollection &) {
  // TODO
  return mlir::success();
}

mlir::LogicalResult WriteFieldOp::verifySymbolUses(mlir::SymbolTableCollection &) {
  // TODO
  return mlir::success();
}

namespace {

LogicalResult verifyGlobalName(SymbolTableCollection &tables, Operation *op, SymbolRefAttr name) {
  ModuleOp mod = op->getParentOfType<ModuleOp>();
  assert(mod);
  GlobalDefOp def = tables.lookupSymbolIn<GlobalDefOp>(mod, name);
  if (!def) {
    return op->emitOpError() << "reference to undefined global '" << name << "'";
  }
  return success();
}

} // namespace

LogicalResult GetGlobalOp::verifySymbolUses(SymbolTableCollection &tables) {
  return verifyGlobalName(tables, *this, getNameRef());
}

LogicalResult SetGlobalOp::verifySymbolUses(SymbolTableCollection &tables) {
  return verifyGlobalName(tables, *this, getNameRef());
}

mlir::OpFoldResult LitValOp::fold(LitValOp::FoldAdaptor) { return getValueAttr(); }

mlir::OpFoldResult LitValArrayOp::fold(LitValArrayOp::FoldAdaptor) { return getElementsAttr(); }

mlir::OpFoldResult ValToIndexOp::fold(ValToIndexOp::FoldAdaptor adaptor) {
  return adaptor.getVal();
}

OpFoldResult GetArrayLenOp::fold(GetArrayLenOp::FoldAdaptor) {
  if (auto compType = dyn_cast<ComponentType>(getArray().getType())) {
    auto arrSize = compType.getArraySize();
    if (failed(arrSize)) {
      return nullptr;
    }
    if (isa<IntegerAttr>(*arrSize)) {
      return *arrSize;
    }
  }
  return nullptr;
}

LogicalResult NopOp::fold(NopOp::FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult> &results) {
  if (adaptor.getIns().size() != getNumResults()) {
    return failure();
  }

  bool atLeastOne = false;
  for (auto in : adaptor.getIns()) {
    if (in) {
      atLeastOne = true;
      results.push_back(in);
    } else {
      results.push_back(nullptr);
    }
  }
  return success(atLeastOne);
}

LogicalResult NopOp::canonicalize(NopOp op, PatternRewriter &rewriter) {
  if (op.getIns().size() == op.getNumResults()) {
    rewriter.replaceAllUsesWith(op.getResults(), op.getIns());
    return success();
  }
  return failure();
}

static bool isTypeVar(Type t) { return mlir::isa<TypeVarType>(t); }

LogicalResult SuperCoerceOp::verify() {
  ComponentType inputType = getComponent().getType();
  Type outputType = getVal().getType();

  if (isTypeVar(outputType)) {
    return success();
  }
  if (!inputType.subtypeOf(outputType)) {
    return emitError() << "type " << outputType << " is not a valid super type of type "
                       << inputType;
  }
  return success();
}

} // namespace zml

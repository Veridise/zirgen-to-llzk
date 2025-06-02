//===- Helpers.cpp - Conversion helpers -------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <functional>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <zklang/Dialect/LLZK/Builder.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Params.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/OpInterfaces.h>
#include <zklang/Dialect/ZML/IR/Ops.h>
#include <zklang/Dialect/ZML/Typing/Materialize.h>
#include <zklang/Dialect/ZML/Utils/Helpers.h>

using namespace mlir;
using namespace zhl;
using namespace zml;

/// Extracts the binding from the input value/operation and creates a
/// cast of the value into the type materialized from the binding.
/// If the super type is specified generates an additional SuperCoerceOp
/// from the binding's type to the super type.
/// REQUIRES that the operation has one result only.
mlir::FailureOr<mlir::Value> zml::CastHelper::getCastedValue(
    mlir::Operation *op, mlir::OpBuilder &builder, mlir::SmallVectorImpl<mlir::Operation *> &,
    mlir::Type super
) {
  assert(op->getNumResults() == 1 && "casting helper can only work with operations with 1 result");
  return getCastedValue(op->getResult(0), builder, super);
}

mlir::FailureOr<mlir::Value>
zml::CastHelper::getCastedValue(mlir::Operation *op, mlir::OpBuilder &builder, mlir::Type super) {
  mlir::SmallVector<mlir::Operation *, 2> genOps;
  return getCastedValue(op, builder, genOps, super);
}

mlir::FailureOr<mlir::Value> zml::CastHelper::getCastedValue(
    mlir::Value value, mlir::OpBuilder &builder,
    mlir::SmallVector<mlir::Operation *, 2> &generatedOps, mlir::Type super
) {
  auto binding = zml::TypeBindingAttr::get(value);
  if (!binding) {
    return mlir::failure();
  }
  return getCastedValue(value, *binding, builder, generatedOps, super);
}

mlir::FailureOr<mlir::Value>
zml::CastHelper::getCastedValue(mlir::Value value, mlir::OpBuilder &builder, mlir::Type super) {
  auto binding = zml::TypeBindingAttr::get(value);
  if (!binding) {
    return mlir::failure();
  }
  mlir::SmallVector<mlir::Operation *, 2> generatedOps;
  return getCastedValue(value, *binding, builder, generatedOps, super);
}

/// A non-failing version that takes a binding as additional parameter
mlir::Value zml::CastHelper::getCastedValue(
    mlir::Value value, const zhl::TypeBinding &binding, mlir::OpBuilder &builder,
    mlir::SmallVectorImpl<mlir::Operation *> &generatedOps, mlir::Type super
) {
  auto materialized = materializeTypeBinding(builder.getContext(), binding);
  assert(materialized);
  if (value.getType() == materialized) {
    return value;
  }
  auto cast = builder.create<mlir::UnrealizedConversionCastOp>(
      value.getLoc(), mlir::TypeRange(materialized), mlir::ValueRange(value)
  );
  generatedOps.push_back(cast.getOperation());

  mlir::Value result = cast.getResult(0);
  if (super && super != materialized) {
    auto coerce = builder.create<zml::SuperCoerceOp>(value.getLoc(), super, result);
    generatedOps.push_back(coerce.getOperation());
    result = coerce;
  }
  return result;
}

mlir::Value zml::CastHelper::getCastedValue(
    mlir::Value value, const zhl::TypeBinding &binding, mlir::OpBuilder &builder, mlir::Type super
) {
  mlir::SmallVector<mlir::Operation *, 2> generatedOps;
  return getCastedValue(value, binding, builder, generatedOps, super);
}

mlir::Operation *zml::findCallee(mlir::StringRef name, mlir::ModuleOp root) {
  auto calleeComp = root.lookupSymbol<llzk::component::StructDefOp>(name);
  if (calleeComp) {
    return calleeComp;
  }

  // Zhl Component ops don't declare their symbols in the symbol table
  // for (auto zhlOp : root.getOps<zirgen::Zhl::ComponentOp>()) {
  //   if (zhlOp.getName() == name) {
  //     return zhlOp;
  //   }
  // }
  return nullptr;
}

template <typename T, typename Fn> T queryComp(mlir::Operation *op, Fn query, T Default = T()) {
  if (auto zmlOp = mlir::dyn_cast<zml::ComponentInterface>(op)) {
    return query(zmlOp);
  }
  return Default;
}

bool zml::calleeIsBuiltin(mlir::Operation *op) {
  return queryComp<bool>(op, [](auto zmlOp) { return zmlOp.getBuiltin(); });
}

#define DEBUG_TYPE "zml-slot-helpers"

FlatSymbolRefAttr zml::createSlot(
    zhl::ComponentSlot *slot, OpBuilder &builder, llzk::component::StructDefOp component,
    Location loc, const TypeConverter &tc
) {
  mlir::SymbolTable st(component);

  LLVM_DEBUG(
      llvm::dbgs() << "Creating a slot in " << component.getName()
                   << "\nDesired name: " << slot->getSlotName() << "\n"
  );
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(&component.getRegion().front().front());
  auto type = tc.convertType(materializeTypeBinding(builder.getContext(), slot->getBinding()));
  auto fieldDef =
      builder.create<llzk::component::FieldDefOp>(loc, slot->getSlotName(), type, slot->isColumn());

  LLVM_DEBUG(llvm::dbgs() << "Field op: " << fieldDef << "\n");
  // Insert the FieldDefOp into the symbol table to make sure it has an unique name within the
  // component
  auto name = st.insert(fieldDef);
  assert(name == slot->getSlotName() && "slot names should be uniqued during type checking");
  LLVM_DEBUG(llvm::dbgs() << "Name created by the symbol table: " << name << "\n");
  if (slot->getSlotName() != name) {
    slot->rename(name.getValue());
  }
  return mlir::FlatSymbolRefAttr::get(component.getContext(), name);
}

static TypeBinding unwrapArrayNTimes(const TypeBinding &type, size_t count) {
  if (count == 0) {
    return type;
  }
  assert(type.isArray());
  auto inner = type.getArrayElement([]() { return mlir::InFlightDiagnostic(); });
  assert(succeeded(inner));
  return unwrapArrayNTimes(*inner, count - 1);
}

Value zml::storeAndLoadSlot(
    ComponentSlot &slot, Value value, FlatSymbolRefAttr slotName, Type slotType, Location loc,
    OpBuilder &builder, Value self
) {
  storeSlot(slot, value, slotName, slotType, loc, builder, self);
  return loadSlot(slot, value.getType(), slotName, slotType, loc, builder, self);
}

Value zml::loadSlot(
    ComponentSlot &slot, Type valueType, FlatSymbolRefAttr slotName, Type slotType, Location loc,
    OpBuilder &builder, Value self
) {
  auto compSlotBinding = slot.getBinding();
  auto compSlotIVs = slot.collectIVs();

  if (compSlotIVs.empty()) {
    // Read the temporary back to a SSA value
    return builder.create<ReadFieldOp>(loc, slotType, self, slotName);
  } else {
    // Read the array back to a SSA value
    auto arrayDataBis = builder.create<ReadFieldOp>(loc, slotType, self, slotName);

    // Read the value we wrote into the array back to a SSA value
    return builder.create<ReadArrayOp>(loc, valueType, arrayDataBis, compSlotIVs);
  }
}

void zml::storeSlot(
    ComponentSlot &slot, Value value, FlatSymbolRefAttr slotName, Type slotType, Location loc,
    OpBuilder &builder, Value self
) {
  auto compSlotBinding = slot.getBinding();
  auto compSlotIVs = slot.collectIVs();

  if (compSlotIVs.empty()) {
    LLVM_DEBUG(
        llvm::dbgs() << "slotType == " << slotType << " | value.getType() == " << value.getType()
                     << "\n"
    );
    assert(slotType == value.getType() && "result of construction and slot type must be the same");
    // Write the construction in a temporary
    builder.create<WriteFieldOp>(loc, self, slotName, value);
  } else {
    auto unwrappedBinding = unwrapArrayNTimes(compSlotBinding, compSlotIVs.size());
    auto unwrappedType = materializeTypeBinding(builder.getContext(), unwrappedBinding);
    assert(
        unwrappedType == value.getType() &&
        "result of construction and slot inner array type must be the same"
    );
    // Read the array from the slot field
    auto arrayData = builder.create<ReadFieldOp>(loc, slotType, self, slotName);
    assert(arrayData);
    assert(value);
    // Write into the array the value
    builder.create<WriteArrayOp>(loc, arrayData, compSlotIVs, value, true);

    // Write the array back into the field
    builder.create<WriteFieldOp>(loc, self, slotName, arrayData);
  }
}

#undef DEBUG_TYPE

static void materializeFieldTypes(
    const zhl::TypeBinding &binding, MLIRContext *ctx, SmallVectorImpl<Type> &types,
    SmallVectorImpl<StringRef> &fields, SmallVectorImpl<bool> *columns = nullptr
) {
  llvm::StringMap<std::pair<Type, bool>> map;
  SmallVector<StringRef> fieldsToSort;
  fieldsToSort.reserve(binding.getMembers().size());

  for (auto &[memberName, memberBinding] : binding.getMembers()) {
    assert(memberBinding.has_value());
    auto memberType = materializeTypeBinding(ctx, *memberBinding);
    bool isColumn = false;
    if (auto *cslot = mlir::dyn_cast_if_present<ComponentSlot>(memberBinding->getSlot())) {
      isColumn = cslot->isColumn();
    }
    map[memberName] = {memberType, isColumn};
    fieldsToSort.push_back(memberName);
  }

  std::sort(fieldsToSort.begin(), fieldsToSort.end());
  for (auto field : fieldsToSort) {
    types.push_back(map[field].first);
    if (columns) {
      columns->push_back(map[field].second);
    }
  }
  fields.insert(fields.end(), fieldsToSort.begin(), fieldsToSort.end());
}

static void constructFieldReads(
    TypeRange types, ArrayRef<StringRef> fieldNames, SmallVectorImpl<Value> &results,
    OpBuilder &builder, Value self, Location loc, const zhl::TypeBinding &binding
) {
  for (auto [type, field] : llvm::zip_equal(types, fieldNames)) {
    auto memberBinding = binding.getMembers().at(field);
    assert(memberBinding.has_value());
    ComponentSlot *slot = mlir::dyn_cast_if_present<ComponentSlot>(memberBinding->getSlot());
    assert(slot);
    auto slotName = FlatSymbolRefAttr::get(builder.getStringAttr(slot->getSlotName()));
    auto slotType = materializeTypeBinding(builder.getContext(), slot->getBinding());
    results.push_back(loadSlot(*slot, type, slotName, slotType, loc, builder, self));
  }
}

FailureOr<Value> zml::constructPODComponent(
    Operation *op, const zhl::TypeBinding &binding, OpBuilder &builder, Value self,
    llvm::function_ref<mlir::Value()> superTypeValueCb, const zhl::TypeBindings &bindings,
    const TypeConverter &tc
) {
  auto ctor = CtorCallBuilder::Make(op, binding, builder, self, bindings);
  if (failed(ctor)) {
    return failure();
  }

  auto loc = binding.getLocation();

  auto superTypeValue = superTypeValueCb();
  SmallVector<Type, 1> argTypes({superTypeValue.getType()});
  SmallVector<StringRef, 1> fieldNames({"$super"});
  SmallVector<Value, 1> args({superTypeValue});
  auto toReserve = 1 + binding.getMembers().size();
  argTypes.reserve(toReserve);
  fieldNames.reserve(toReserve);
  args.reserve(toReserve);

  materializeFieldTypes(binding, builder.getContext(), argTypes, fieldNames);
  constructFieldReads(
      ArrayRef(argTypes).drop_front(), ArrayRef(fieldNames).drop_front(), args, builder, self, loc,
      binding
  );

  return ctor->build(builder, loc, args, tc);
}

void zml::createPODComponent(
    zhl::TypeBinding &binding, mlir::OpBuilder &builder, mlir::SymbolTable &st,
    const mlir::TypeConverter &TC
) {
  SmallVector<Type> argTypes;
  SmallVector<StringRef> fieldNames;
  SmallVector<bool> columns;

  llzk::ComponentBuilder cb;
  auto loc = binding.getLocation();
  cb.name(binding.getName()).isClosure().location(loc);

  auto superType =
      TC.convertType(materializeTypeBinding(builder.getContext(), binding.getSuperType()));
  argTypes = {superType};
  fieldNames = {"$super"};
  columns = {false};
  argTypes.reserve(1 + binding.getMembers().size());
  fieldNames.reserve(1 + binding.getMembers().size());
  columns.reserve(1 + binding.getMembers().size());

  materializeFieldTypes(binding, builder.getContext(), argTypes, fieldNames, &columns);
  for (auto [memberName, memberType, isColumn] : llvm::zip_equal(fieldNames, argTypes, columns)) {
    cb.field(memberName, TC.convertType(memberType), isColumn);
  }

  if (!binding.getGenericParamNames().empty()) {
    cb.forceGeneric().typeParams(binding.getGenericParamNames());
  }
  cb.defer([&](auto compOp) {
    auto actualName = st.insert(compOp);
    binding.setName(actualName.getValue());

    // Now that we know the final name of the component we can configure the rest
    cb.fillBody(
        llvm::map_to_vector(argTypes, [&TC](Type t) { return TC.convertType(t); }),
        {TC.convertType(materializeTypeBinding(builder.getContext(), binding))},
        [&](mlir::ValueRange args, mlir::OpBuilder &B, const TypeConverter &tc) {
      auto componentType = tc.convertType(materializeTypeBinding(B.getContext(), binding));
      // Reference to self
      auto self = B.create<llzk::component::CreateStructOp>(loc, componentType);
      // Store the fields
      for (auto [fieldName, arg] : llvm::zip_equal(fieldNames, args)) {
        B.create<llzk::component::FieldWriteOp>(loc, self, fieldName, arg);
      }
      // Return self
      B.create<llzk::function::ReturnOp>(loc, mlir::ValueRange({self}));
    }
    );
  });
  auto compOp = cb.build(builder, TC);
  assert(compOp);
}

mlir::FailureOr<CtorCallBuilder> CtorCallBuilder::Make(
    mlir::Operation *op, mlir::Value value, const zhl::ZIRTypeAnalysis &typeAnalysis,
    mlir::OpBuilder &builder, mlir::Value self, const zhl::TypeBindings &bindings
) {
  auto binding = typeAnalysis.getType(value);
  if (failed(binding)) {
    return op->emitError() << "failed to type check";
  }

  return Make(op, *binding, builder, self, bindings);
}

mlir::FailureOr<CtorCallBuilder> CtorCallBuilder::Make(
    mlir::Operation *op, const zhl::TypeBinding &binding, mlir::OpBuilder &builder,
    mlir::Value self, const zhl::TypeBindings &bindings
) {
  auto rootModule = op->getParentOfType<mlir::ModuleOp>();
  auto *calleeComp = findCallee(binding.getName(), rootModule);
  if (!calleeComp) {
    return op->emitError() << "could not find component with name " << binding.getName();
  }
  auto constructorType = materializeTypeBindingConstructor(builder, binding, bindings);

  return CtorCallBuilder(
      constructorType, binding, op->getParentOfType<llzk::component::StructDefOp>(), self,
      calleeIsBuiltin(calleeComp)
  );
}

mlir::Value CtorCallBuilder::build(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args, const TypeConverter &tc,
    GlobalBuilder::AndName globalBuilder
) {
  auto genericParams = compBinding.getGenericParamsMapping();
  auto ref = builder.create<ConstructorRefOp>(
      loc, compBinding.getName(), genericParams.size() - genericParams.sizeOfDeclared(), ctorType,
      isBuiltin
  );

  auto call = builder.create<mlir::func::CallIndirectOp>(loc, ref, args);
  Value compValue = call.getResult(0);

  if (FrameSlot *slot = compBinding.getSlot()) {
    assert(!globalBuilder && "cannot be both field and global");
    zhl::ComponentSlot *compSlot = mlir::cast<zhl::ComponentSlot>(slot);
    TypeBinding compSlotBinding = compSlot->getBinding();

    // Create the field
    FlatSymbolRefAttr name = createSlot(compSlot, builder, callerComponentOp, loc, tc);
    Type slotType = materializeTypeBinding(builder.getContext(), compSlotBinding);

    compValue = storeAndLoadSlot(*compSlot, compValue, name, slotType, loc, builder, self);
  } else if (globalBuilder) {
    SymbolRefAttr name = GlobalBuilder::buildDef(*globalBuilder, loc, compValue.getType());
    builder.create<SetGlobalOp>(loc, name, compValue);
    compValue = builder.create<GetGlobalOp>(loc, compValue.getType(), name);
  }

  builder.create<ConstrainCallOp>(loc, compValue, args);
  return compValue;
}

mlir::FunctionType CtorCallBuilder::getCtorType() const { return ctorType; }

const zhl::TypeBinding &CtorCallBuilder::getBinding() const { return compBinding; }

CtorCallBuilder::CtorCallBuilder(
    mlir::FunctionType type, const zhl::TypeBinding &binding, llzk::component::StructDefOp caller,
    mlir::Value selfValue, bool builtin
)
    : ctorType(type), compBinding(binding), isBuiltin(builtin), callerComponentOp(caller),
      self(selfValue) {
  assert(callerComponentOp);
  assert(self);
}

mlir::FailureOr<Value> zml::coerceToArray(TypedValue<ComponentLike> v, OpBuilder &builder) {
  auto arraySuper = v.getType().getFirstMatchingSuperType([](Type t) {
    if (auto ct = mlir::dyn_cast_if_present<ComponentLike>(t)) {
      return ct.isConcreteArray();
    }
    return false;
  });

  if (failed(arraySuper)) {
    return failure();
  }

  return builder.create<SuperCoerceOp>(v.getLoc(), *arraySuper, v).getResult();
}

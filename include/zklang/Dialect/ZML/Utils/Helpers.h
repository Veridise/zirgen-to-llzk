//===- Helpers.h - Conversion helpers ---------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes utility functions and classes that simplify or reduce
// boilerplate in the ZHL->ZML conversion patterns.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <mlir/Transforms/DialectConversion.h>
#include <vector>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZHL/Typing/Analysis.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZML/IR/Ops.h>

namespace zml {

/// Finds the definition of the callee component. If the
/// component was defined before the current operation w.r.t. the physical order of
/// the file then it's defined by a ZML ComponentInterface op, if it hasn't been
/// converted yet it is still a ZHL Component op. If the name could not be found
/// in either form returns nullptr.
mlir::Operation *findCallee(mlir::StringRef name, mlir::ModuleOp root);

/// Returns true if the operation is a component and has the builtin attribute
bool calleeIsBuiltin(mlir::Operation *op);

/// Creates a slot for storing a result inside a component's body.
mlir::FlatSymbolRefAttr createSlot(
    zhl::ComponentSlot *slot, mlir::OpBuilder &builder, ComponentInterface component,
    mlir::Location loc
);

/// Stores a value into the field defined by the slotName symbol and immediately reads it back.
/// Returns the value read from the field.
mlir::Value storeAndLoadSlot(
    zhl::ComponentSlot &slot, mlir::Value value, mlir::FlatSymbolRefAttr slotName,
    mlir::Type slotType, mlir::Location loc, mlir::OpBuilder &builder, mlir::Value
);

/// Returns the value read from a field created from a slot.
mlir::Value loadSlot(
    zhl::ComponentSlot &slot, mlir::Type valueType, mlir::FlatSymbolRefAttr slotName,
    mlir::Type slotType, mlir::Location loc, mlir::OpBuilder &builder, mlir::Value
);

/// Stores a value into the field defined by the slotName symbol.
void storeSlot(
    zhl::ComponentSlot &slot, mlir::Value value, mlir::FlatSymbolRefAttr slotName,
    mlir::Type slotType, mlir::Location loc, mlir::OpBuilder &builder, mlir::Value
);

/// Given a Value of type ComponentType it returns a value of type Array<T,N> where
/// said array is one of the super types of the input's type.
mlir::FailureOr<mlir::Value> coerceToArray(mlir::TypedValue<ComponentType> v, mlir::OpBuilder &);

/// Calls the constructor of the POD component associated with the binding using the provided super
/// type value. The rest of the fields of the POD component are read from the actual component using
/// the slots associated with the member's bindings.
///
/// The super type's value is provided with a lazy callback that will get called iff the function
/// knowns it will succeed. This makes safe to create operations and other conversion modifications
/// for obtaining the value of the super type.
mlir::FailureOr<mlir::Value>
constructPODComponent(mlir::Operation *op, zhl::TypeBinding &binding, mlir::OpBuilder &builder, mlir::Value self, llvm::function_ref<mlir::Value()> superTypeValueCb, const zhl::TypeBindings &);

/// Creates a component used to represent a closure. This component will have a constructor that
/// takes as input the values for all fields including the super type's value.
void createPODComponent(zhl::TypeBinding &, mlir::OpBuilder &, mlir::SymbolTable &);

/// Helper for constructing global variables in a common global module.
class GlobalBuilder {
  mlir::ModuleOp &globalsModuleRef;

public:
  GlobalBuilder(mlir::ModuleOp &globalsModule) : globalsModuleRef(globalsModule) {}

  mlir::SymbolRefAttr buildName(mlir::StringAttr flatName) const {
    return mlir::SymbolRefAttr::get(
        globalsModuleRef.getSymNameAttr(), {mlir::FlatSymbolRefAttr::get(flatName)}
    );
  }

  void buildDef(mlir::Location loc, mlir::StringAttr flatName, mlir::Type type) const {
    // This must be called for both Zhl::ConstructGlobalOp and Zhl::GetGlobalOp (because there's no
    // requirement that a global is defined within the current compilation) so check if it already
    // exists before creating the GlobalDefOp.
    if (globalsModuleRef.lookupSymbol(flatName) == nullptr) {
      mlir::OpBuilder bldr(globalsModuleRef.getRegion());
      bldr.create<GlobalDefOp>(loc, flatName, mlir::TypeAttr::get(type));
    }
  }

  using AndName = std::optional<std::pair<const GlobalBuilder *, mlir::StringAttr>>;

  inline AndName forName(mlir::StringAttr nameOfGlobal) const {
    return std::make_pair(this, nameOfGlobal);
  }

  /// Build `GlobalDefOp` and return its full, unambiguous name.
  inline static mlir::SymbolRefAttr
  buildDef(const AndName &builder, mlir::Location loc, mlir::Type type) {
    mlir::StringAttr name = builder->second;
    builder->first->buildDef(loc, name, type);
    return builder->first->buildName(name);
  }
};

/// Helper for creating the ops that represent the call to a component's constructor.
/// If the associated binding is linked to a frame slot it also creates the field and writes the
/// result into it.
class CtorCallBuilder {
public:
  static mlir::FailureOr<CtorCallBuilder> Make(
      mlir::Operation *op, mlir::Value value, const zhl::ZIRTypeAnalysis &typeAnalysis,
      mlir::OpBuilder &builder, mlir::Value self, const zhl::TypeBindings &bindings
  );

  static mlir::FailureOr<CtorCallBuilder> Make(
      mlir::Operation *op, const zhl::TypeBinding &binding, mlir::OpBuilder &builder,
      mlir::Value self, const zhl::TypeBindings &bindings
  );

  /// Generates the ops that represent the construction of a component. Fetches a reference to
  /// the constructor and calls it. The constructed component is also used as argument to create
  /// a constrain call op that represents the call to `@constrain` in LLZK. Returns the argument to
  /// the constrain call op (which may be the direct result of the construction or the result of a
  /// write-read round trip described below which is needed to break the def-use chain of the
  /// returned value from the component construction so the latter can be removed in the LLZK
  /// `@constrain` functions).
  ///
  /// If the binding is linked to a frame slot, it also generates a field in the component and
  /// generates the following between the construction and the constrain call op:
  ///   - a field write storing the constructed value to the new field
  ///   - a field read retrieving the constructed value from the new field
  ///
  /// Otherwise, if a `GlobalBuilder::AndName` instance is provided (i.e. the call is handling a
  /// `zhl.construct_global` op), generates a global declaration op in the "globals" module and
  /// generates the following between the construction and the constrain call op:
  ///   - a global write storing the constructed value to the new global
  ///   - a global read retrieving the constructed value from the new global
  mlir::Value build(
      mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args,
      GlobalBuilder::AndName globalBuilder = std::nullopt
  );
  mlir::FunctionType getCtorType() const;
  const zhl::TypeBinding &getBinding() const;

private:
  CtorCallBuilder(
      mlir::FunctionType type, const zhl::TypeBinding &binding, ComponentInterface caller,
      mlir::Value self, bool builtin
  );

  mlir::Value
  buildCallWithoutSlot(mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args);

  mlir::FunctionType ctorType;
  const zhl::TypeBinding compBinding;
  bool isBuiltin;
  ComponentInterface callerComponentOp;
  mlir::Value self;
};

} // namespace zml

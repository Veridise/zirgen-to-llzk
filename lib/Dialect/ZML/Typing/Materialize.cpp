//===- Materialize.cpp - Type binding to MLIR type --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <iterator>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <unordered_set>
#include <zklang/Dialect/ZHL/Typing/Params.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/Specialization.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>
#include <zklang/Dialect/ZML/BuiltIns/BuiltIns.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/Types.h>
#include <zklang/Dialect/ZML/Typing/Materialize.h>

#define DEBUG_TYPE "zml-type-materialization"

using namespace mlir;
using namespace zhl;
using namespace zml;

static void assertValidSuperType(Type superType) {
  auto validSuperType = mlir::isa<ComponentLike, TypeVarType>(superType);
  (void)validSuperType;
  assert(validSuperType && "supertype is not a component type or a type variable");
}

namespace {

static std::unordered_set<std::string_view> feltEquivalentTypes(
    {"Val", "Add", "Sub", "Mul", "BitAnd", "Inv", "Isz", "InRange", "Neg", "Mod"}
),
    extValBuiltins({"ExtVal", "ExtAdd", "ExtSub", "ExtMul", "ExtInv", "MakeExt"});

class Materializer {
public:
  explicit Materializer(MLIRContext *ctx) : context(ctx) {}

  Type materializeTypeBinding(const TypeBinding &binding) {
    seen.clear();
    return materializeImpl(binding);
  }

private:
  Type materializeImpl(const TypeBinding &binding) {
    checkCycle(binding);

    auto base = materializeBaseType(binding);

    seen.pop_back();
    if (binding.isVariadic()) {
      return VarArgsType::get(context, base);
    }
    return base;
  }

  SmallVector<Attribute> materializeGenericParamNames(const TypeBinding &binding) {
    auto names = binding.getGenericParamNames();
    return llvm::map_to_vector(names, [&](const auto &name) -> Attribute {
      return SymbolRefAttr::get(StringAttr::get(context, name));
    });
  }

  Attribute materializeAttribute(const TypeBinding &binding) {
    // Special case for constants that do not have a known value
    if (binding.isConst() && !binding.isKnownConst()) {
      auto intAttr = IntegerAttr::get(IntegerType::get(context, 64), ShapedType::kDynamic);
      LLVM_DEBUG(
          llvm::dbgs() << "<== Materializing " << binding << " to IntegerAttr: " << intAttr << "\n"
      );
      return intAttr;
    }
    if (binding.isGenericParam() && !binding.getSuperType().isTypeMarker()) {
      Attribute symAttr =
          FlatSymbolRefAttr::get(StringAttr::get(context, binding.getGenericParamName()));
      if (binding.getSuperType().hasConstExpr()) {
        if (auto constExprAttr =
                mlir::dyn_cast<ConstExprAttr>(materializeAttribute(binding.getSuperType()))) {
          symAttr = LiftedExprAttr::get(mlir::dyn_cast<FlatSymbolRefAttr>(symAttr), constExprAttr);
        }
      }

      LLVM_DEBUG(
          llvm::dbgs() << "<== Materializing " << binding << " to FlatSymbolRefAttr: " << symAttr
                       << "\n"
      );
      return symAttr;
    }
    if (binding.hasConstExpr()) {
      Builder builder(context);
      auto attrResult = binding.getConstExpr().convertIntoAttribute(builder);
      LLVM_DEBUG(
          llvm::dbgs() << "<== Materializing " << binding << " to Attribute: " << attrResult << "\n"
      );
      return attrResult;
    }

    auto typeAttr = TypeAttr::get(materializeImpl(binding));
    LLVM_DEBUG(
        llvm::dbgs() << "<== Materializing " << binding << " to TypeAttr: " << typeAttr << "\n"
    );
    return typeAttr;
  }

  Type materializeGenericType(const TypeBinding &binding, Type superType) {
    if (!binding.isSpecialized()) {
      auto params = materializeGenericParamNames(binding);
      return ComplexComponentType::get(
          context, binding.getName(), mlir::cast<ComponentLike>(superType), params,
          binding.isBuiltin()
      );
    }

    auto paramBindings = binding.getGenericParams();
    SmallVector<Attribute, 2> params =
        map_to_vector(paramBindings, std::bind_front(&Materializer::materializeAttribute, this));

    return ComplexComponentType::get(
        context, binding.getName(), mlir::cast<ComponentLike>(superType), params,
        binding.isBuiltin()
    );
  }

  Type materializeBaseType(const TypeBinding &binding) {
    if (!binding.hasSuperType() || binding.isTypeMarker()) {
      return builtins::Component(context);
    }
    if (binding.isConst()) {
      return builtins::Val(context);
    }

    if (binding.isGenericParam()) {
      if (binding.getSuperType().isTypeMarker()) {
        LLVM_DEBUG(llvm::dbgs() << "<== Materializing " << binding << " to TypeVarType\n");
        return TypeVarType::get(
            context, SymbolRefAttr::get(StringAttr::get(context, binding.getGenericParamName()))
        );
      }
      if (binding.getSuperType().isVal() || binding.getSuperType().isTransitivelyVal()) {
        LLVM_DEBUG(llvm::dbgs() << "<== Materializing " << binding << " to Val\n");
        return builtins::Val(context);
      }

      assert(false && "Generic param that is neither Val or Type");
    }
    auto superType = materializeImpl(binding.getSuperType());
    assertValidSuperType(superType);

    if (binding.isGeneric()) {
      return materializeGenericType(binding, superType);
    }

    // Convert Val and ExtVal and other builtins into llzk types here.
    if (feltEquivalentTypes.contains(binding.getName())) {
      return builtins::Val(context);
    }
    if (extValBuiltins.contains(binding.getName())) {
      return builtins::ExtVal(context);
    }
    if (binding.getName() == "String") {
      return builtins::String(context);
    }

    return ComplexComponentType::get(
        context, binding.getName(), mlir::cast<ComponentLike>(superType), binding.isBuiltin()
    );
  }

  void checkCycle(const TypeBinding &binding) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    binding.print(ss);
    std::unordered_set set(seen.begin(), seen.end());
    assert(set.find(s) == set.end() && "cycle detected");
    seen.push_back(s);
  }

  MLIRContext *context;
  std::vector<std::string> seen;
};

} // namespace

static Type materializeTypeBindingImpl(MLIRContext *context, const TypeBinding &binding) {
  Materializer m(context);
  return m.materializeTypeBinding(binding);
}

Type zml::materializeTypeBinding(MLIRContext *context, const TypeBinding &binding) {
  return materializeTypeBindingImpl(context, binding);
}

/// Materializes a type binding after replacing generic parameters that are in scope with the actual
/// instantiated type.
static Type specializeAndMaterializeTypeBinding(
    MLIRContext *ctx, const TypeBinding &binding, const Params &scope, const TypeBindings &bindings
) {
  // If the scope is empty or the binding we are specializing is not generic then there are no type
  // variables.
  if (scope.empty() || !binding.isGeneric()) {
    LLVM_DEBUG(llvm::dbgs() << binding << ": no specialization needed\n");
    return materializeTypeBinding(ctx, binding);
  }

  // Make a copy to assign the params to
  auto copy = binding;
  ParamsScopeStack scopeStack(scope);

  if (failed(zhl::specializeTypeBinding(&copy, scopeStack, bindings))) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to specialize binding " << binding << "\n");
    return nullptr;
  }

  return materializeTypeBindingImpl(ctx, copy);
}

#ifndef NDEBUG
static llvm::raw_ostream &indent(size_t count = 1) {
  for (size_t i = 0; i < count; i++) {
    llvm::dbgs() << "|";
    llvm::dbgs().indent(2);
  }
  return llvm::dbgs();
}
#endif

FunctionType zml::materializeTypeBindingConstructor(
    Builder &builder, const TypeBinding &binding, const TypeBindings &bindings
) {
  auto genericParams = binding.getGenericParamsMapping();
  LLVM_DEBUG(llvm::dbgs() << "Materializing constructor type for " << binding << "\n");
  // Create the type of the binding and of each argument
  // then return a function type using the generated types.
  // If any of the given types is a null just return nullptr for the whole thing.
  std::vector<Type> args;
  auto retType =
      specializeAndMaterializeTypeBinding(builder.getContext(), binding, genericParams, bindings);
  if (!retType) {
    LLVM_DEBUG(indent() << "failed to materialize the return type for " << binding << "\n");
    return nullptr;
  }
  LLVM_DEBUG(indent() << "Materialized return type to " << retType << "\n");

  LLVM_DEBUG(indent() << "For binding " << binding << " constructor types are: \n");
  auto params = binding.getConstructorParams();
  std::transform(params.begin(), params.end(), std::back_inserter(args), [&](auto &argBinding) {
    LLVM_DEBUG(indent(2) << argBinding << "\n");
    auto materializedType = specializeAndMaterializeTypeBinding(
        builder.getContext(), argBinding, genericParams, bindings
    );
    LLVM_DEBUG(indent() << "Materialized to " << materializedType << "\n");
    return materializedType;
  });
  if (std::any_of(args.begin(), args.end(), [](Type t) { return !t; })) {
    LLVM_DEBUG(indent() << "Failed to materialize an argument type for " << binding << "\n");
    return nullptr;
  }

  auto ctorType = builder.getFunctionType(args, retType);
  LLVM_DEBUG(
      llvm::dbgs() << "<== Materializing constructor function of " << binding << " to " << ctorType
                   << "\n"
  );
  return ctorType;
}

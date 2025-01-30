#include "zklang/Dialect/ZML/Typing/Materialize.h"
#include "zklang/Dialect/ZHL/Typing/Specialization.h"
#include "zklang/Dialect/ZHL/Typing/TypeBindings.h"
#include "zklang/Dialect/ZML/IR/Types.h"
#include <algorithm>
#include <iterator>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <unordered_set>

#define DEBUG_TYPE "zml-type-materialization"

using namespace mlir;
using namespace zhl;

namespace zkc::Zmir {

void errMsg(const TypeBinding &binding, StringRef reason) {
  LLVM_DEBUG(
      llvm::dbgs() << "failed to materialize type for " << binding << ": " << reason << "\n"
  );
}

namespace {

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

  Type materializeBaseType(const TypeBinding &binding) {
    if (!binding.hasSuperType() || binding.isTypeMarker()) {
      return ComponentType::Component(context);
    }
    if (binding.isGenericParam() && binding.getSuperType().isTypeMarker()) {
      return TypeVarType::get(
          context, SymbolRefAttr::get(StringAttr::get(context, binding.getGenericParamName()))
      );
    }
    if (binding.isGenericParam() && binding.getSuperType().isVal()) {
      return ComponentType::Val(context);
    }
    auto superType = materializeImpl(binding.getSuperType());
    if (!mlir::isa<ComponentType, TypeVarType>(superType)) {
      errMsg(binding, "supertype is not a component type or a type variable");
      return nullptr;
    }
    if (binding.isGeneric()) {
      std::vector<Attribute> params;
      if (binding.isSpecialized()) {
        // Put the types associated with the specialization
        auto paramBindings = binding.getGenericParams();
        std::transform(
            paramBindings.begin(), paramBindings.end(), std::back_inserter(params),
            [&](const auto &b) -> Attribute {
              if (b.isConst()) {
                return mlir::IntegerAttr::get(
                    mlir::IntegerType::get(context, 64),
                    b.isKnownConst() ? b.getConst() : mlir::ShapedType::kDynamic
                );
              } else if (b.isGenericParam() && b.getSuperType().isVal()) {
                return SymbolRefAttr::get(StringAttr::get(context, b.getGenericParamName()));
              } else {
                return mlir::TypeAttr::get(materializeImpl(b));
              }
            }
        );
      } else {
        // Put the names of the parameters
        auto names = binding.getGenericParamNames();
        std::transform(
            names.begin(), names.end(), std::back_inserter(params),
            [&](const auto &name) { return SymbolRefAttr::get(StringAttr::get(context, name)); }
        );
      }

      return ComponentType::get(context, binding.getName(), superType, params, binding.isBuiltin());
    } else if (binding.isConst()) {
      return ComponentType::Val(context);
    }
    return ComponentType::get(context, binding.getName(), superType, binding.isBuiltin());
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

Type materializeTypeBinding(MLIRContext *context, const TypeBinding &binding) {
  Materializer m(context);
  return m.materializeTypeBinding(binding);
}

/// Materializes a type binding after replacing generic parameters that are in scope with the actual
/// instantiated type.
Type specializeAndMaterializeTypeBinding(
    MLIRContext *ctx, const TypeBinding &binding, const Params &scope
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
  auto result = zhl::specializeTypeBinding(&copy, scopeStack);
  if (failed(result)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to specialize binding " << binding << "\n");
    return nullptr;
  }

  return materializeTypeBinding(ctx, copy);
}

FunctionType materializeTypeBindingConstructor(OpBuilder &builder, const TypeBinding &binding) {
  auto &genericParams = binding.getGenericParamsMapping();
  // Create the type of the binding and of each argument
  // then return a function type using the generated types.
  // If any of the given types is a null just return nullptr for the whole thing.
  std::vector<Type> args;
  auto retType = specializeAndMaterializeTypeBinding(builder.getContext(), binding, genericParams);
  if (!retType) {
    LLVM_DEBUG(llvm::dbgs() << "failed to materialize the return type for " << binding << "\n");
    return nullptr;
  }

  LLVM_DEBUG(llvm::dbgs() << "For binding " << binding << " constructor types are: \n");
  auto &params = binding.getConstructorParams();
  std::transform(params.begin(), params.end(), std::back_inserter(args), [&](auto &argBinding) {
    LLVM_DEBUG(llvm::dbgs() << "|  " << argBinding << "\n");
    auto materializedType =
        specializeAndMaterializeTypeBinding(builder.getContext(), argBinding, genericParams);
    LLVM_DEBUG(llvm::dbgs() << "Materialized to " << materializedType << "\n");
    return materializedType;
  });
  if (std::any_of(args.begin(), args.end(), [](Type t) { return !t; })) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to materialize an argument type for " << binding << "\n");
    return nullptr;
  }

  return builder.getFunctionType(args, retType);
}

} // namespace zkc::Zmir

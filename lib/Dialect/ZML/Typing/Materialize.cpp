#include <algorithm>
#include <iterator>
#include <llvm/Support/Debug.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <unordered_set>
#include <zklang/Dialect/ZHL/Typing/Params.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/Specialization.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>
#include <zklang/Dialect/ZML/IR/Types.h>
#include <zklang/Dialect/ZML/Typing/Materialize.h>

#define DEBUG_TYPE "zml-type-materialization"

using namespace mlir;
using namespace zhl;
using namespace zml;

static void errMsg(const TypeBinding &binding, StringRef reason) {
  LLVM_DEBUG(
      llvm::dbgs() << "failed to materialize type for " << binding << ": " << reason << "\n"
  );
}

static void assertValidSuperType(Type superType) {
  auto validSuperType = mlir::isa<ComponentType, TypeVarType>(superType);
  (void)validSuperType;
  assert(validSuperType && "supertype is not a component type or a type variable");
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

  SmallVector<Attribute, 2> materializeGenericParamNames(const TypeBinding &binding) {
    SmallVector<Attribute, 2> params;
    auto names = binding.getGenericParamNames();
    params.reserve(names.size());
    std::transform(names.begin(), names.end(), std::back_inserter(params), [&](const auto &name) {
      return SymbolRefAttr::get(StringAttr::get(context, name));
    });
    return params;
  }

  Attribute materializeAttribute(const TypeBinding &binding) {
    // Special case for constants that do not have a known value
    if (binding.isConst() && !binding.isKnownConst()) {
      auto intAttr =
          mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), mlir::ShapedType::kDynamic);
      LLVM_DEBUG(
          llvm::dbgs() << "<== Materializing " << binding << " to IntegerAttr: " << intAttr << "\n"
      );
      return intAttr;
    }

    if (binding.hasConstExpr()) {
      mlir::Builder builder(context);
      auto attrResult = binding.getConstExpr().convertIntoAttribute(builder);

      LLVM_DEBUG(
          llvm::dbgs() << "<== Materializing " << binding << " to Attribute: " << attrResult << "\n"
      );
      return attrResult;
    }

    if (binding.isGenericParam()) {
      auto symAttr =
          FlatSymbolRefAttr::get(StringAttr::get(context, binding.getGenericParamName()));

      LLVM_DEBUG(
          llvm::dbgs() << "<== Materializing " << binding << " to FlatSymbolRefAttr: " << symAttr
                       << "\n"
      );
      return symAttr;
    }

    auto typeAttr = mlir::TypeAttr::get(materializeImpl(binding));
    LLVM_DEBUG(
        llvm::dbgs() << "<== Materializing " << binding << " to TypeAttr: " << typeAttr << "\n"
    );
    return typeAttr;
  }

  Type materializeGenericType(const TypeBinding &binding, Type superType) {
    if (!binding.isSpecialized()) {
      auto params = materializeGenericParamNames(binding);
      return ComponentType::get(context, binding.getName(), superType, params, binding.isBuiltin());
    }

    auto paramBindings = binding.getGenericParams();
    SmallVector<Attribute, 2> params;
    params.reserve(paramBindings.size());
    std::transform(
        paramBindings.begin(), paramBindings.end(), std::back_inserter(params),
        std::bind(&Materializer::materializeAttribute, this, std::placeholders::_1)
    );

    return ComponentType::get(context, binding.getName(), superType, params, binding.isBuiltin());
  }

  Type materializeBaseType(const TypeBinding &binding) {
    if (!binding.hasSuperType() || binding.isTypeMarker()) {
      return ComponentType::Component(context);
    }
    if (binding.isConst()) {
      return ComponentType::Val(context);
    }

    if (binding.isGenericParam()) {
      if (binding.getSuperType().isTypeMarker()) {
        LLVM_DEBUG(llvm::dbgs() << "<== Materializing " << binding << " to TypeVarType\n");
        return TypeVarType::get(
            context, SymbolRefAttr::get(StringAttr::get(context, binding.getGenericParamName()))
        );
      }
      if (binding.getSuperType().isVal()) {
        LLVM_DEBUG(llvm::dbgs() << "<== Materializing " << binding << " to Val\n");
        return ComponentType::Val(context);
      }
      if (binding.getSuperType().isTransitivelyVal()) {
        LLVM_DEBUG(llvm::dbgs() << "<== Materializing " << binding << " to Val\n");
        return ComponentType::Val(context);
      }
      assert(false && "Generic param that is neither Val or Type");
    }
    auto superType = materializeImpl(binding.getSuperType());
    assertValidSuperType(superType);

    if (binding.isGeneric()) {
      return materializeGenericType(binding, superType);
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
  LogicalResult result = failure();
  ParamsScopeStack scopeStack(scope);
  result = zhl::specializeTypeBinding(&copy, scopeStack);

  if (failed(result)) {
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

FunctionType
zml::materializeTypeBindingConstructor(OpBuilder &builder, const TypeBinding &binding) {
  auto genericParams = binding.getGenericParamsMapping();
  LLVM_DEBUG(llvm::dbgs() << "Materializing constructor type for " << binding << "\n");
  // Create the type of the binding and of each argument
  // then return a function type using the generated types.
  // If any of the given types is a null just return nullptr for the whole thing.
  std::vector<Type> args;
  auto retType = specializeAndMaterializeTypeBinding(builder.getContext(), binding, genericParams);
  if (!retType) {
    LLVM_DEBUG(indent() << "failed to materialize the return type for " << binding << "\n");
    return nullptr;
  }
  LLVM_DEBUG(indent() << "Materialized return type to " << retType << "\n");

  LLVM_DEBUG(indent() << "For binding " << binding << " constructor types are: \n");
  auto params = binding.getConstructorParams();
  std::transform(params.begin(), params.end(), std::back_inserter(args), [&](auto &argBinding) {
    LLVM_DEBUG(indent(2) << argBinding << "\n");
    auto materializedType =
        specializeAndMaterializeTypeBinding(builder.getContext(), argBinding, genericParams);
    LLVM_DEBUG(indent() << "Materialized to " << materializedType << "\n");
    return materializedType;
  });
  if (std::any_of(args.begin(), args.end(), [](Type t) { return !t; })) {
    LLVM_DEBUG(indent() << "Failed to materialize an argument type for " << binding << "\n");
    return nullptr;
  }

  return builder.getFunctionType(args, retType);
}

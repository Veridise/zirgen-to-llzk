#include "zklang/Dialect/ZML/Typing/Materialize.h"
#include "zklang/Dialect/ZHL/Typing/TypeBindings.h"
#include "zklang/Dialect/ZML/IR/Types.h"
#include <algorithm>
#include <iterator>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <unordered_set>

using namespace mlir;
using namespace zhl;

namespace zkc::Zmir {

static int errIdent;
void errMsg(const TypeBinding &binding, StringRef reason) {
  for (int x = 0; x < errIdent; x++) {
    llvm::dbgs() << "  ";
  }
  llvm::dbgs() << "failed to materialize type for ";
  binding.print(llvm::dbgs());
  llvm::dbgs() << ": " << reason << "\n";
}

Type inner_materializeTypeBinding(MLIRContext *context, const TypeBinding &binding);

static std::vector<std::string> seen;

Type materializeTypeBinding(MLIRContext *context, const TypeBinding &binding) {
  seen.clear();
  errIdent = 0;
  errMsg(binding, "clearing seen set");
  return inner_materializeTypeBinding(context, binding);
}

Type inner_materializeTypeBinding(MLIRContext *context, const TypeBinding &binding) {
  {
    errIdent++;
    std::string s;
    llvm::raw_string_ostream ss(s);
    binding.print(ss);
    llvm::dbgs() << "              - bind: " << s << "\n";
    for (auto &se : seen) {
      llvm::dbgs() << "              - seen: " << se << "\n";
    }
    std::unordered_set set(seen.begin(), seen.end());
    assert(set.find(s) == set.end() && "cycle detected");
    seen.push_back(s);
  }
  errMsg(binding, "entering scope");
  if (binding.isGenericParam()) {
    errMsg(binding, "is generic param");
  }
  if (binding.isBottom()) {
    errMsg(binding, "binding is the bottom type");
    return nullptr;
  }

  /*if (binding.isTypeMarker()) {*/
  /*  errMsg(binding, "binding is a marker type");*/
  /*}*/
  /*assert(!binding.isTypeMarker() && "Cannot materialize the type marker");*/

  auto materializeBaseType = [&]() -> mlir::Type {
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
    auto superType = inner_materializeTypeBinding(context, binding.getSuperType());
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
            return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), b.getConst());
          } else if (b.isGenericParam() && b.getSuperType().isVal()) {
            return SymbolRefAttr::get(StringAttr::get(context, b.getGenericParamName()));
          } else {
            return mlir::TypeAttr::get(inner_materializeTypeBinding(context, b));
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

      return ComponentType::get(context, binding.getName(), superType, params);
    } else if (binding.isConst()) {
      return ComponentType::Val(context);
    } else {
      return ComponentType::get(context, binding.getName(), superType);
    }
  };

  auto result = binding.isVariadic() ? VarArgsType::get(context, materializeBaseType())
                                     : materializeBaseType();

  errMsg(binding, "exiting scope");
  errIdent--;
  seen.pop_back();
  return result;
}

void spaces(size_t n) {
  for (size_t i = 0; i < n; i++) {
    llvm::dbgs() << "|  ";
  }
}

class ParamsScopeStack {
public:
  explicit ParamsScopeStack(const Params &root) { stack.push_back(&root); }
  const TypeBinding *operator[](std::string_view name) {
    for (auto It = stack.rbegin(); It != stack.rend(); ++It) {
      auto level = *It;
      auto binding = (*level)[name];
      if (binding != nullptr) {
        return binding;
      }
    }
    return nullptr;
  }

  void pushScope(const Params &param) { stack.push_back(&param); }

  void popScope() { stack.pop_back(); }

  void print(llvm::raw_ostream &os) {

    os << "[Scope top -> ";
    for (auto It = stack.rbegin(); It != stack.rend(); ++It) {
      os << *It << " -> ";
    }
    os << " <- bottom]\n";
  }

private:
  std::vector<const Params *> stack;
};

class ScopeGuard {
public:
  explicit ScopeGuard(ParamsScopeStack &stack, const Params &param) : stack(&stack) {
    stack.pushScope(param);
  }
  ~ScopeGuard() { stack->popScope(); }

private:
  ParamsScopeStack *stack;
};

LogicalResult
specializeTypeBinding(MLIRContext *ctx, TypeBinding *dst, ParamsScopeStack &scopes, size_t ident) {
  spaces(ident);
  scopes.print(llvm::dbgs());
  // If the destionation is null do nothing
  if (dst == nullptr) {
    return success();
  }
  // If the type binding is a generic param replace it with the actual type.
  if (dst->isGenericParam()) {
    spaces(ident);
    llvm::dbgs() << "Specializing ";
    dst->print(llvm::dbgs());
    llvm::dbgs() << "\n";
    auto varName = dst->getGenericParamName();
    auto replacement = scopes[varName];
    if (replacement == nullptr) {
      spaces(ident);
      llvm::dbgs() << "Failed to convert because " << varName << " was not found\n";
      return failure();
    }
    spaces(ident);
    llvm::dbgs() << "Potential replacement ";
    replacement->print(llvm::dbgs());
    llvm::dbgs() << "\n";
    // If the replacement is the type marker don't do anything
    if (replacement->isTypeMarker()) {
      spaces(ident);
      llvm::dbgs() << "Not replaced because there is no actual type to replace with\n";
      return success();
    }
    // If the variable's replacement is a Val that is not constant then we don't do anything.
    if (!replacement->isConst() && replacement->isVal() && dst->getSuperType().isVal()) {
      spaces(ident);
      llvm::dbgs() << "Not replaced because there is no actual value to replace with\n";
      return success();
    }
    // Create a copy of the replacement in the destination
    *dst = *replacement;
    spaces(ident);
    llvm::dbgs() << "Into ";
    dst->print(llvm::dbgs());
    llvm::dbgs() << "  (generic)\n";
    return success();
  }
  // If the type is a generic type apply the replacement to its generic parameters
  if (dst->isGeneric()) {
    auto &params = dst->getGenericParamsMapping();
    for (auto &name : dst->getGenericParamNames()) {
      if (!params[name]->isGenericParam()) {
        continue;
      }
      spaces(ident);
      llvm::dbgs() << "Variable " << name << " binds to ";
      params[name]->print(llvm::dbgs());
      llvm::dbgs() << "\n";
      spaces(ident);
      llvm::dbgs() << "Specializing variable " << name;
      auto replacement = scopes[name];
      // Fail the materialization if the name was not found. A well typed program should not have
      // this issue.
      if (replacement == nullptr) {
        llvm::dbgs() << " failed to convert because " << name << " was not found\n";
        return failure();
      }

      // If the replacement is the type marker don't do anything
      if (replacement->isTypeMarker()) {
        llvm::dbgs() << " was not replaced because there is no actual type to replace with\n";
        continue;
      }

      // If the variable's replacement is a Val that is not constant then we don't do anything.
      // This applies to the variables whose parent is of Val type
      if (!replacement->isConst() && replacement->isVal() && params[name]->getSuperType().isVal()) {
        llvm::dbgs() << " was not replaced because there is no actual value to replace with\n";
        continue;
      }
      // Create a copy of the actual type
      auto copy = *replacement;
      llvm::dbgs() << " replaced with ";
      copy.print(llvm::dbgs());
      llvm::dbgs() << "\n";
      // And specialize it
      /*auto result = success();*/
      auto &newScope = copy.getGenericParamsMapping();
      {
        ScopeGuard guard(scopes, newScope);
        auto result = specializeTypeBinding(ctx, &copy, scopes, ident + 1);
        if (failed(result)) {
          spaces(ident);
          llvm::dbgs() << "Failure\n";
          return failure();
        }
      }
      spaces(ident);
      llvm::dbgs() << "Into ";
      copy.print(llvm::dbgs());
      llvm::dbgs() << "  (param)\n";
      // Replace the type binding with the specialization we just generated.
      dst->replaceGenericParamByName(name, copy);
    }
    // After the replacement follow the chain on super types and apply specializations.
    if (dst->hasSuperType()) {
      auto &superTypeScope = dst->getSuperType().getGenericParamsMapping();
      spaces(ident);
      llvm::dbgs() << "Specializing super type ";
      dst->getSuperType().print(llvm::dbgs());
      llvm::dbgs() << "\n";
      {
        TypeBinding copy = dst->getSuperType();
        ScopeGuard guard(scopes, superTypeScope);
        auto superTypeResult = specializeTypeBinding(ctx, &copy, scopes, ident + 1);
        if (failed(superTypeResult)) {
          spaces(ident);
          llvm::dbgs() << "Failure\n";
          return failure();
        }
        if (!copy.isTypeMarker()) {
          dst->getSuperType() = copy;
        }
      }
      spaces(ident);
      llvm::dbgs() << "Into ";
      dst->getSuperType().print(llvm::dbgs());
      llvm::dbgs() << "  (super type)\n";
    }
    return success();
  }

  // Do nothing in the other cases.
  spaces(ident);
  llvm::dbgs() << "Nothing\n";
  return success();
}

/// Materializes a type binding after replacing generic parameters that are in scope with the actual
/// instantiated type.
Type specializeAndMaterializeTypeBinding(
    MLIRContext *ctx, const TypeBinding &binding, const Params &scope
) {
  // If the scope is empty or the binding we are specializing is not generic then there are no type
  // variables.
  if (scope.empty() || !binding.isGeneric()) {
    errMsg(binding, "no specialization needed");
    return materializeTypeBinding(ctx, binding);
  }

  errMsg(binding, "specializing");
  // Make a copy to assign the params to
  auto copy = binding;
  ParamsScopeStack scopeStack(scope);
  auto result = specializeTypeBinding(ctx, &copy, scopeStack, 2);
  if (failed(result)) {
    return nullptr;
  }

  return materializeTypeBinding(ctx, copy);
}

FunctionType materializeTypeBindingConstructor(OpBuilder &builder, const TypeBinding &binding) {
  errMsg(binding, "entering materializing context");
  auto &genericParams = binding.getGenericParamsMapping();
  // Create the type of the binding and of each argument
  // then return a function type using the generated types.
  // If any of the given types is a null just return nullptr for the whole thing.
  std::vector<Type> args;
  auto retType = specializeAndMaterializeTypeBinding(builder.getContext(), binding, genericParams);
  if (!retType) {
    llvm::dbgs() << "failed to materialize the return type for ";
    binding.print(llvm::dbgs());
    llvm::dbgs() << "\n";
    return nullptr;
  }

  llvm::dbgs() << "For binding ";
  binding.print(llvm::dbgs());
  llvm::dbgs() << " constructor types are: \n";
  auto &params = binding.getConstructorParams();
  std::transform(params.begin(), params.end(), std::back_inserter(args), [&](auto &argBinding) {
    llvm::dbgs() << "|  ";
    argBinding.print(llvm::dbgs());
    llvm::dbgs() << "\n";
    auto materializedType =
        specializeAndMaterializeTypeBinding(builder.getContext(), argBinding, genericParams);
    llvm::dbgs() << "Materialized to " << materializedType << "\n";
    return materializedType;
  });
  if (std::any_of(args.begin(), args.end(), [](Type t) { return !t; })) {
    llvm::dbgs() << "failed to materialize an argument type for ";
    binding.print(llvm::dbgs());
    llvm::dbgs() << "\n";
    return nullptr;
  }

  errMsg(binding, "exiting materializing context");
  return builder.getFunctionType(args, retType);
}

} // namespace zkc::Zmir

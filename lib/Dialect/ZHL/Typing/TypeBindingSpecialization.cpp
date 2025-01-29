#include "zklang/Dialect/ZHL/Typing/TypeBindings.h"
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/Debug.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace zhl;

// Copied from Materialize.cpp
// TODO: Unify both implementations if specialization is still necessary in Materialize.cpp

// TODO: Use LLVM debug mechanism, but that requires a Debug build of LLVM
#define ENABLE_DEBUG
#ifdef ENABLE_DEBUG
#define _LLVM_DEBUG(X) {X}
#else
#define _LLVM_DEBUG(X)
#endif

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
      (*It)->printMapping(os);
      os << " -> ";
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

LogicalResult specializeTypeBinding(
    TypeBinding *dst, ParamsScopeStack &scopes, const llvm::StringSet<> &FV, size_t ident
);

inline LogicalResult specializeTypeBinding_genericParamCase(
    TypeBinding *dst, ParamsScopeStack &scopes, const llvm::StringSet<> &FV, size_t ident
) {
  _LLVM_DEBUG(spaces(ident); llvm::dbgs() << "Specializing " << *dst << "\n";)
  auto varName = dst->getGenericParamName();
  if (FV.contains(varName)) {
    _LLVM_DEBUG(spaces(ident); llvm::dbgs() << "Is a free variable\n";)
    return success();
  }
  auto replacement = scopes[varName];
  if (replacement == nullptr) {
    _LLVM_DEBUG(spaces(ident);
                llvm::dbgs() << "Failed to convert because " << varName << " was not found\n";)
    return failure();
  }
  _LLVM_DEBUG(spaces(ident); llvm::dbgs() << "Potential replacement " << *replacement << "\n";)
  // If the replacement is the type marker don't do anything
  if (replacement->isTypeMarker()) {
    _LLVM_DEBUG(spaces(ident);
                llvm::dbgs() << "Not replaced because there is no actual type to replace with\n";)
    return success();
  }
  // If the variable's replacement is a Val that is not constant then we don't do anything.
  if (!replacement->isConst() && replacement->isVal() && dst->getSuperType().isVal()) {
    _LLVM_DEBUG(spaces(ident);
                llvm::dbgs() << "Not replaced because there is no actual value to replace with\n";)
    return success();
  }
  // Create a copy of the replacement in the destination
  *dst = *replacement;
  _LLVM_DEBUG(spaces(ident); llvm::dbgs() << "Into " << *dst << "  (generic)\n";)
  return success();
}

inline LogicalResult specializeTypeBinding_genericTypeCase(
    TypeBinding *dst, ParamsScopeStack &scopes, const llvm::StringSet<> &FV, size_t ident
) {
  auto &params = dst->getGenericParamsMapping();
  for (auto &name : dst->getGenericParamNames()) {
    if (!params[name]->isGenericParam()) {
      continue;
    }
    _LLVM_DEBUG(spaces(ident);
                llvm::dbgs() << "Variable " << name << " binds to " << *params[name] << "\n";
                spaces(ident); llvm::dbgs() << "Specializing variable " << name;)
    auto replacement = scopes[name];
    // Fail the materialization if the name was not found. A well typed program should not have
    // this issue.
    if (replacement == nullptr) {
      _LLVM_DEBUG(llvm::dbgs() << " failed to convert because " << name << " was not found\n";)
      return failure();
    }

    // If the replacement is the type marker don't do anything
    if (replacement->isTypeMarker()) {
      _LLVM_DEBUG(llvm::dbgs(
                  ) << " was not replaced because there is no actual type to replace with\n";)
      continue;
    }

    // If the variable's replacement is a Val that is not constant then we don't do anything.
    // This applies to the variables whose parent is of Val type
    if (!replacement->isConst() && replacement->isVal() && params[name]->getSuperType().isVal()) {
      _LLVM_DEBUG(llvm::dbgs(
                  ) << " was not replaced because there is no actual value to replace with\n";)
      continue;
    }
    // Create a copy of the actual type
    auto copy = *replacement;
    _LLVM_DEBUG(llvm::dbgs() << " replaced with " << copy << "\n";)
    // And specialize it if necessary
    auto &newScope = copy.getGenericParamsMapping();
    {
      ScopeGuard guard(scopes, newScope);
      auto result = specializeTypeBinding(&copy, scopes, FV, ident + 1);
      if (failed(result)) {
        _LLVM_DEBUG(spaces(ident); llvm::dbgs() << "Failure\n";)
        return failure();
      }
    }
    _LLVM_DEBUG(spaces(ident); llvm::dbgs() << "Into " << copy << "  (param)\n";)
    // Replace the type binding with the specialization we just generated.
    dst->replaceGenericParamByName(name, copy);
  }
  // After the replacement follow the chain on super types and apply specializations.
  if (dst->hasSuperType()) {
    auto &superTypeScope = dst->getSuperType().getGenericParamsMapping();
    _LLVM_DEBUG(spaces(ident);
                llvm::dbgs() << "Specializing super type " << dst->getSuperType() << "\n";)
    TypeBinding copy = dst->getSuperType();
    {
      ScopeGuard guard(scopes, superTypeScope);
      auto superTypeResult = specializeTypeBinding(&copy, scopes, FV, ident + 1);
      if (failed(superTypeResult)) {
        _LLVM_DEBUG(spaces(ident); llvm::dbgs() << "Failure\n";)
        return failure();
      }
      if (copy.isGeneric()) {
        copy.markAsSpecialized();
      }
      if (!copy.isTypeMarker()) {
        dst->getSuperType() = copy;
      }
    }
    _LLVM_DEBUG(spaces(ident); llvm::dbgs() << "Into " << dst->getSuperType() << "  (super type)\n";
    )
  }
  // Specialize the types of the constructor's arguments
  auto &constructorParams = dst->getConstructorParams();
  for (auto &param : constructorParams) {
    auto &paramScope = param.getGenericParamsMapping();
    _LLVM_DEBUG(spaces(ident);
                llvm::dbgs() << "Specializing constructor argument's type " << param << "\n";);
    {
      ScopeGuard guard(scopes, paramScope);
      auto paramTypeResult = specializeTypeBinding(&param, scopes, FV, ident + 1);
      if (failed(paramTypeResult)) {
        _LLVM_DEBUG(spaces(ident); llvm::dbgs() << "Failure\n";)
        return failure();
      }
    }
  }
  // Specialize the types of the members
  auto &members = dst->getMembers();
  for (auto &[name, type] : members) {
    if (type.has_value()) {
      auto &memberScope = type->getGenericParamsMapping();
      _LLVM_DEBUG(spaces(ident);
                  llvm::dbgs() << "Specializing member " << name << " of type " << *type << "\n";);
      {
        ScopeGuard guard(scopes, memberScope);
        if (failed(specializeTypeBinding(&type.value(), scopes, FV, ident + 1))) {
          _LLVM_DEBUG(spaces(ident); llvm::dbgs() << "Failure\n";)
          return failure();
        }
      }
    } else {
      _LLVM_DEBUG(spaces(ident);
                  llvm::dbgs() << "Ignored " << name << " because it does not have a type";)
    }
  }

  return success();
}

void printFVs(const llvm::StringSet<> &FV, llvm::raw_ostream &os) {
  size_t c = 1;
  os << "{ ";
  for (auto name : FV.keys()) {
    os << name;
    if (c < FV.size()) {
      os << ", ";
    }
    c++;
  }
  os << " }\n";
}

LogicalResult specializeTypeBinding(
    TypeBinding *dst, ParamsScopeStack &scopes, const llvm::StringSet<> &FV, size_t ident
) {
  _LLVM_DEBUG(spaces(ident); scopes.print(llvm::dbgs()); spaces(ident); printFVs(FV, llvm::dbgs());)
  // If the destionation is null do nothing
  if (dst == nullptr) {
    return success();
  }
  // If the type binding is a generic param replace it with the actual type.
  if (dst->isGenericParam()) {
    return specializeTypeBinding_genericParamCase(dst, scopes, FV, ident);
  }
  // If the type is a generic type apply the replacement to its generic parameters
  if (dst->isGeneric()) {
    return specializeTypeBinding_genericTypeCase(dst, scopes, FV, ident);
  }

  // Do nothing in the other cases.
  _LLVM_DEBUG(spaces(ident); llvm::dbgs() << "Nothing\n";)
  return success();
}

mlir::FailureOr<zhl::TypeBinding> zhl::TypeBinding::specialize(
    std::function<mlir::InFlightDiagnostic()> emitError, mlir::ArrayRef<TypeBinding> params
) const {
  if (specialized) {
    return emitError() << "can't respecialize type '" << getName() << "'";
  }
  if (genericParams.size() == 0) {
    return emitError() << "type '" << name << "' is not generic";
  }
  if (genericParams.size() != params.size()) {
    return emitError() << "wrong number of specialization parameters. Expected "
                       << genericParams.size() << " but got " << params.size();
  }

  // Root cause, the specialization does not update all the places where the type could reference a
  // generic. I could port the specialization logic I implemented in the materializer here and make
  // it use this implementation.
  ParamsMap generics; // The root scope for specialization
  llvm::StringSet<>
      freeVariables; // Any type variable introduced by the specialization is a free variable.
  for (unsigned i = 0; i < params.size(); i++) {
    // TODO: Validation
    generics.insert({{genericParams.getName(i), i}, params[i]});
    if (params[i].isGenericParam()) {
      freeVariables.insert(params[i].getGenericParamName());
    }
  }

  // TypeBinding specializedBinding(name, loc, *superType, generics, constructorParams, members);
  TypeBinding specializedBinding(*this); // Make a copy to create the specialization

  Params initialScope(generics);
  ParamsScopeStack scopeStack(initialScope);
  auto result = specializeTypeBinding(&specializedBinding, scopeStack, freeVariables, 2);
  if (failed(result)) {
    _LLVM_DEBUG(llvm::dbgs() << "Failed to specialize binding " << *this << "\n";)
    return failure();
  }

  specializedBinding.markAsSpecialized();
  return specializedBinding;
}

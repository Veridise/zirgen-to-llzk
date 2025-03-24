#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/Debug.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/Expr.h>
#include <zklang/Dialect/ZHL/Typing/Params.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/Specialization.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

#include <llvm/ADT/TypeSwitch.h>

// Field used for constant folding operations. Selected for feature parity with zirgen.
#include <zklang/FiniteFields/BabyBear.h>

using namespace mlir;
using namespace zhl;
using namespace zhl::expr;

#define DEBUG_TYPE "zhl-type-binding-specialization"

void spaces(size_t n) {
  for (size_t i = 0; i < n; i++) {
    llvm::dbgs() << "|  ";
  }
}

ParamsScopeStack::ParamsScopeStack(const Params &root) { stack.push_back(&root); }

const TypeBinding *ParamsScopeStack::operator[](StringRef name) {
  StringRef currName = name;
  const TypeBinding *result = nullptr;
  for (auto It = stack.rbegin(); It != stack.rend(); ++It) {
    auto level = *It;
    auto binding = (*level)[currName];
    if (binding != nullptr) {
      result = binding;
      if (binding->isGenericParam()) {
        currName = binding->getGenericParamName();
      }
    }
  }

  return result;
}

void ParamsScopeStack::pushScope(const Params &param) { stack.push_back(&param); }

void ParamsScopeStack::popScope() { stack.pop_back(); }

void ParamsScopeStack::print(llvm::raw_ostream &os) const {
  os << "[Scope top -> ";
  for (auto It = stack.rbegin(); It != stack.rend(); ++It) {
    (*It)->printMapping(os);
    os << " -> ";
  }
  os << " <- bottom]\n";
}

namespace {

class ScopeGuard {
public:
  explicit ScopeGuard(ParamsScopeStack &Stack, const Params &param) : stack(&Stack) {
    stack->pushScope(param);
  }
  ~ScopeGuard() { stack->popScope(); }

private:
  ParamsScopeStack *stack;
};

} // namespace

static LogicalResult specializeTypeBindingImpl(
    TypeBinding *dst, ParamsScopeStack &scopes, const llvm::StringSet<> &FV,
    const TypeBindings &bindings, size_t indent
);

static LogicalResult specializeTypeBinding_genericParamCase(
    TypeBinding *dst, ParamsScopeStack &scopes, const llvm::StringSet<> &FV,
    [[maybe_unused]] const TypeBindings &bindings, size_t indent
) {
  LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Specializing " << *dst << "\n");
  auto varName = dst->getGenericParamName();
  if (FV.contains(varName)) {
    LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Is a free variable\n");
    return success();
  }
  auto replacement = scopes[varName];
  if (replacement == nullptr) {
    LLVM_DEBUG(spaces(indent);
               llvm::dbgs() << "Failed to convert because " << varName << " was not found\n");
    return failure();
  }
  LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Potential replacement " << *replacement << "\n");
  // If the replacement is the type marker don't do anything
  if (replacement->isTypeMarker()) {
    LLVM_DEBUG(spaces(indent);
               llvm::dbgs() << "Not replaced because there is no actual type to replace with\n");
    return success();
  }
  // If the variable's replacement is a Val that is not constant then we don't do anything.
  if (!replacement->isConst() && replacement->isVal() && dst->getSuperType().isVal()) {
    LLVM_DEBUG(spaces(indent);
               llvm::dbgs() << "Not replaced because there is no actual value to replace with\n");
    return success();
  }
  // Create a copy of the replacement in the destination
  *dst = *replacement;
  LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Into " << *dst << "  (generic)\n");
  return success();
}

static LogicalResult specializeTypeBinding_genericTypeCase(
    TypeBinding *dst, ParamsScopeStack &scopes, const llvm::StringSet<> &FV,
    const TypeBindings &bindings, size_t indent
) {
  auto params = dst->getGenericParamsMapping();
  for (auto &name : dst->getGenericParamNames()) {

    LLVM_DEBUG(spaces(indent);
               llvm::dbgs() << "Variable '" << name << "' binds to '" << *params[name] << "'\n");
    if (!params[name]->isGenericParam()) {
      LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Specializing " << *params[name] << "\n");
      auto result = specializeTypeBindingImpl(params[name], scopes, FV, bindings, indent + 1);
      if (failed(result)) {
        LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Failure\n");
        return failure();
      }

      continue;
    }
    LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Specializing variable '"
                                            << params[name]->getGenericParamName() << "'");
    auto replacement = scopes[params[name]->getGenericParamName()];
    // Fail the materialization if the name was not found. A well typed program should not have
    // this issue.
    if (replacement == nullptr) {
      LLVM_DEBUG(
          llvm::dbgs() << " failed to convert because '" << *params[name] << "' was not found\n"
      );
      return failure();
    }

    // If the replacement is the type marker don't do anything
    if (replacement->isTypeMarker()) {
      LLVM_DEBUG(
          llvm::dbgs() << " was not replaced because there is no actual type to replace with\n"
      );
      continue;
    }

    // If the variable's replacement is a Val that is not constant then we don't do anything.
    // This applies to the variables whose parent is of Val type
    if (!replacement->isConst() && replacement->isVal() && params[name]->getSuperType().isVal()) {
      LLVM_DEBUG(
          llvm::dbgs() << " was not replaced because there is no actual value to replace with\n"
      );
      continue;
    }
    // Create a copy of the actual type
    auto copy = *replacement;
    LLVM_DEBUG(llvm::dbgs() << " replaced with " << copy << "\n");
    // And specialize it if necessary
    if (failed(specializeTypeBindingImpl(&copy, scopes, FV, bindings, indent + 1))) {
      LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Failure\n");
      return failure();
    }
    LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Into " << copy << "  (param)\n");
    // Replace the type binding with the specialization we just generated.
    dst->replaceGenericParamByName(name, copy);
  }
  // After the replacement follow the chain on super types and apply specializations.
  if (dst->hasSuperType()) {
    auto superTypeScope = dst->getSuperType().getGenericParamsMapping();
    LLVM_DEBUG(spaces(indent);
               llvm::dbgs() << "Specializing super type " << dst->getSuperType() << "\n");
    TypeBinding copy = dst->getSuperType();
    ScopeGuard guard(scopes, superTypeScope);
    if (failed(specializeTypeBindingImpl(&copy, scopes, FV, bindings, indent + 1))) {
      LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Failure\n");
      return failure();
    }
    if (copy.isGeneric()) {
      copy.markAsSpecialized();
    }
    if (!copy.isTypeMarker()) {
      dst->setSuperType(bindings.Manage(copy));
    }
    LLVM_DEBUG(spaces(indent);
               llvm::dbgs() << "Into " << dst->getSuperType() << "  (super type)\n");
  }
  // Specialize the types of the constructor's arguments
  auto constructorParams = dst->getConstructorParams();
  for (auto &param : constructorParams) {
    LLVM_DEBUG(spaces(indent);
               llvm::dbgs() << "Specializing constructor argument's type " << param << "\n");
    {
      auto paramTypeResult = specializeTypeBindingImpl(&param, scopes, FV, bindings, indent + 1);
      if (failed(paramTypeResult)) {
        LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Failure\n");
        return failure();
      }
      if (param.isGeneric()) {
        param.markAsSpecialized();
      }
    }
  }
  // Specialize the types of the members
  auto &members = dst->getMembers();
  for (auto &member : members) {
    auto name = member.getKey();
    auto &type = member.getValue();
    if (type.has_value()) {
      auto memberScope = type->getGenericParamsMapping();
      LLVM_DEBUG(spaces(indent);
                 llvm::dbgs() << "Specializing member " << name << " of type " << *type << "\n");
      {
        ScopeGuard guard(scopes, memberScope);
        if (failed(specializeTypeBindingImpl(&type.value(), scopes, FV, bindings, indent + 1))) {
          LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Failure\n");
          return failure();
        }
        if (type->isGeneric()) {
          type->markAsSpecialized();
        }
      }
    } else {
      LLVM_DEBUG(spaces(indent);
                 llvm::dbgs() << "Ignored " << name << " because it does not have a type");
    }
  }

  return success();
}

#ifndef NDEBUG
/// Prints the set of free variables into the output stream
static void printFVs(const llvm::StringSet<> &FV, llvm::raw_ostream &os) {
  if (FV.empty()) {
    os << "No FVs\n";
    return;
  }
  os << "FVs { ";
  llvm::interleaveComma(FV.keys(), os);
  os << " }\n";
}
#endif

static FailureOr<ConstExpr>
constantFoldSymExpr(const SymbolView &expr, ParamsScopeStack &scopes, size_t indent) {
  auto name = expr->getName();
  LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Propagating constants in Sym expr " << expr << "\n");
  auto *replacement = scopes[name];
  if (replacement == nullptr) {
    LLVM_DEBUG(spaces(indent + 1); llvm::dbgs() << "Symbol '" << name << "' is not in scope\n");
    return expr;
  }
  if (replacement->isKnownConst()) {
    LLVM_DEBUG(spaces(indent + 1);
               llvm::dbgs() << "Sym expr " << expr << " replaced with " << *replacement << "\n");
    return ConstExpr::Val(replacement->getConst());
  }

  LLVM_DEBUG(spaces(indent + 1); llvm::dbgs() << "Sym expr " << expr << " was not modified\n");
  return expr;
}

static FailureOr<ConstExpr>
constantFoldValExpr(const ValView &expr, [[maybe_unused]] ParamsScopeStack &scopes, size_t indent) {
  LLVM_DEBUG(spaces(indent);
             llvm::dbgs() << "Propagating constants in Val expr " << expr << " is trivial\n");
  return expr;
}

static FailureOr<ConstExpr> constantFoldExpr(const ExprView &, ParamsScopeStack &, size_t);

static llvm::StringSet<> FoldableComponents({"Add", "Sub", "Mul", "Div", "Mod", "Neg"});

static uint64_t foldNeg(uint64_t value) {
  ff::babybear::Field BabyBear;
  return BabyBear.prime - (value % BabyBear.prime);
}

static FailureOr<uint64_t> foldBinaryOp(StringRef name, uint64_t lhs, uint64_t rhs) {
  ff::babybear::Field BabyBear;
  auto fp = [&](uint64_t v) { return v % BabyBear.prime; };
  lhs = fp(lhs);
  rhs = fp(rhs);

  return llvm::StringSwitch<FailureOr<uint64_t>>(name)
      .Case("Add", fp(lhs + rhs))
      .Case("Sub", fp(lhs - rhs))
      .Case("Mul", fp(lhs * rhs))
      .Case("Div", rhs == 0 ? FailureOr<uint64_t>() : fp(lhs / rhs))
      .Case("Mod", rhs == 0 ? FailureOr<uint64_t>() : fp(lhs % rhs))
      .Default(failure());
}

static FailureOr<ConstExpr>
constantFoldCtorExpr(const CtorView &expr, ParamsScopeStack &scopes, size_t indent) {
  auto typeName = expr->getTypeName();

  LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Propagating constants in Ctor expr" << expr << "\n");
  if (!FoldableComponents.contains(typeName)) {
    LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Expr " << expr << " is not foldable\n");
    return expr;
  }

  if (typeName == "Neg") {
    LLVM_DEBUG(spaces(indent + 1); llvm::dbgs() << "Folding Neg constructor\n");
    assert(expr->arguments().size() == 1);

    auto foldedArg = constantFoldExpr(expr->arguments()[0], scopes, indent + 2);
    if (failed(foldedArg)) {
      LLVM_DEBUG(spaces(indent + 1); llvm::dbgs() << "Failed to fold\n");
      return failure();
    }

    LLVM_DEBUG(spaces(indent + 1); llvm::dbgs() << "Folded into " << *foldedArg << "\n");
    if (auto val = mlir::dyn_cast_if_present<ValExpr>(*foldedArg)) {
      LLVM_DEBUG(spaces(indent + 2); llvm::dbgs() << "Argument is a constant value\n");
      auto newValue = foldNeg(val->getValue());
      LLVM_DEBUG(spaces(indent + 1);
                 llvm::dbgs() << "Neg constructor folded into " << newValue << "\n");
      return ConstExpr::Val(newValue);
    }

    LLVM_DEBUG(spaces(indent + 2); llvm::dbgs() << "Argument is not a constant value\n");
    return ConstExpr::Ctor(typeName, {*foldedArg});
  } else {
    LLVM_DEBUG(spaces(indent + 1); llvm::dbgs() << "Folding " << typeName << " constructor\n");
    assert(expr->arguments().size() == 2);

    auto foldedLhs = constantFoldExpr(expr->arguments()[0], scopes, indent + 2);
    if (failed(foldedLhs)) {
      LLVM_DEBUG(spaces(indent + 1); llvm::dbgs() << "Failed to fold left hand side\n");
      return failure();
    }
    LLVM_DEBUG(spaces(indent + 1); llvm::dbgs() << "Folded lhs into " << *foldedLhs << "\n");

    auto foldedRhs = constantFoldExpr(expr->arguments()[1], scopes, indent + 2);
    if (failed(foldedRhs)) {
      LLVM_DEBUG(spaces(indent + 1); llvm::dbgs() << "Failed to fold right hand side\n");
      return failure();
    }

    LLVM_DEBUG(spaces(indent + 1); llvm::dbgs() << "Folded rhs into " << *foldedRhs << "\n");
    auto lhsVal = mlir::dyn_cast_if_present<ValExpr>(*foldedLhs);
    auto rhsVal = mlir::dyn_cast_if_present<ValExpr>(*foldedRhs);
    if (lhsVal && rhsVal) {
      LLVM_DEBUG(spaces(indent + 2); llvm::dbgs() << "Arguments are both constant values\n");
      auto newValue = foldBinaryOp(typeName, lhsVal->getValue(), rhsVal->getValue());
      if (failed(newValue)) {
        LLVM_DEBUG(spaces(indent + 2);
                   llvm::dbgs()
                   << "Failed to fold constructor because the folder returned an invalid value\n");
        return failure();
      }
      LLVM_DEBUG(spaces(indent + 1);
                 llvm::dbgs() << typeName << " constructor folded into " << *newValue << "\n");
      return ConstExpr::Val(*newValue);
    }

    LLVM_DEBUG(spaces(indent + 2); llvm::dbgs() << "Arguments are not both constant values\n");
    return ConstExpr::Ctor(typeName, {*foldedLhs, *foldedRhs});
  }

  LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Unreachable case\n");
  return failure();
}

static FailureOr<ConstExpr>
constantFoldExpr(const ExprView &expr, ParamsScopeStack &scopes, size_t indent) {
  assert(bool(expr) && "Attempted to fold empty ConstExpr");
  LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Folding expression (ConstExpr) " << expr << "\n");
  if (auto val = mlir::dyn_cast<ValView>(expr)) {
    LLVM_DEBUG(spaces(indent); llvm::dbgs() << " Folding ValExpr\n");
    return constantFoldValExpr(val, scopes, indent);
  }
  LLVM_DEBUG(spaces(indent); llvm::dbgs() << " Is not a ValExpr\n");
  if (auto sym = mlir::dyn_cast<SymbolView>(expr)) {
    LLVM_DEBUG(spaces(indent); llvm::dbgs() << " Folding SymExpr\n");
    return constantFoldSymExpr(sym, scopes, indent);
  }
  LLVM_DEBUG(spaces(indent); llvm::dbgs() << " Is not a SymExpr\n");
  if (auto ctor = mlir::dyn_cast<CtorView>(expr)) {
    LLVM_DEBUG(spaces(indent); llvm::dbgs() << " Folding CtorExpr\n");
    return constantFoldCtorExpr(ctor, scopes, indent);
  }
  LLVM_DEBUG(spaces(indent); llvm::dbgs() << " Is not a CtorExpr\n");
  llvm_unreachable("There are no other subclasses of ConstExpr");
  return failure();
}

static Params getConstantParams(const TypeBinding &binding, ParamsStorage &storage) {
  ParamsMap constants;
  Params params = binding.getGenericParamsMapping();
  for (auto [pos, param, name] : llvm::enumerate(params, params.getNames())) {
    if (param.isKnownConst()) {
      constants.declare(name, param, pos);
    }
  }
  storage = ParamsStorage(constants, params.size(), TypeBinding(binding.getLocation()));
  return Params(storage);
}

static LogicalResult propagateConstants(
    TypeBinding *dst, ParamsScopeStack &scopes, const TypeBindings &bindings, size_t indent
) {
  LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Propagating constants for type " << *dst << "\n");

  MutableParams params = dst->getGenericParamsMapping();

  ParamsStorage sto;
  Params constParams = getConstantParams(*dst, sto);
  ScopeGuard guard(scopes, constParams);
  LLVM_DEBUG(spaces(indent); scopes.print(llvm::dbgs()));
  for (size_t param = 0; param < params.size(); param++) {
    auto &paramBinding = params.getParam(param);
    LLVM_DEBUG(spaces(indent + 1); llvm::dbgs() << "Propagating constants in parameter #" << param
                                                << " " << paramBinding << "\n";
               spaces(indent + 2); paramBinding.print(llvm::dbgs() << "Detailed printout: ", true);
               llvm::dbgs() << "\n");

    if (!paramBinding.hasConstExpr()) {
      LLVM_DEBUG(spaces(indent + 1);
                 llvm::dbgs() << "Param " << paramBinding << " has no constant expression\n");
      continue;
    }
    auto folded = constantFoldExpr(paramBinding.getConstExpr(), scopes, indent + 2);
    if (failed(folded)) {
      LLVM_DEBUG(spaces(indent + 1);
                 llvm::dbgs() << "Failed to fold " << paramBinding.getConstExpr() << "\n");
      return failure();
    }
    if (paramBinding.getConstExpr() != *folded) {
      LLVM_DEBUG(spaces(indent + 1); llvm::dbgs()
                                     << "Binding " << paramBinding
                                     << " replaced its constexpr with " << *folded << "\n");
      auto origWasCtor = mlir::isa<CtorExpr>(paramBinding.getConstExpr());
      auto newIsVal = mlir::isa<ValExpr>(*folded);
      if (origWasCtor && newIsVal) {
        // If the expression folded to a constant then extract the value and replace the binding
        // with a Const one.
        paramBinding =
            bindings.Const(mlir::cast<ValExpr>(*folded)->getValue(), paramBinding.getLocation());
      } else {
        paramBinding.setConstExpr(*folded);
      }
    }
  }

  if (dst->hasSuperType()) {
    ParamsStorage superTypeSto;
    Params superTypeScope = getConstantParams(dst->getSuperType(), superTypeSto);
    LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Propagating constants in super type "
                                            << dst->getSuperType() << "\n");
    TypeBinding copy = dst->getSuperType();
    ScopeGuard superTypeGuard(scopes, superTypeScope);
    if (failed(propagateConstants(&copy, scopes, bindings, indent + 1))) {
      LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Failure\n");
      return failure();
    }
    if (!copy.isTypeMarker()) {
      dst->setSuperType(bindings.Manage(copy));
    }
    LLVM_DEBUG(spaces(indent);
               llvm::dbgs() << "Into " << dst->getSuperType() << "  (super type)\n");
  }
  // Specialize the types of the constructor's arguments
  auto constructorParams = dst->getConstructorParams();
  for (auto &param : constructorParams) {
    LLVM_DEBUG(spaces(indent); llvm::dbgs()
                               << "Propagating constants in constructor argument's type " << param
                               << "\n");
    if (failed(propagateConstants(&param, scopes, bindings, indent + 1))) {
      LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Failure\n");
      return failure();
    }
  }
  // Specialize the types of the members
  auto &members = dst->getMembers();
  for (auto &member : members) {
    auto name = member.getKey();
    auto &type = member.getValue();
    if (type.has_value()) {
      ParamsStorage memberSto;
      Params memberScope = getConstantParams(*type, memberSto);
      LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Propagating constants in member " << name
                                              << " of type " << *type << "\n");
      {
        ScopeGuard memberGuard(scopes, memberScope);
        if (failed(propagateConstants(&type.value(), scopes, bindings, indent + 1))) {
          LLVM_DEBUG(spaces(indent); llvm::dbgs() << "Failure\n");
          return failure();
        }
      }
    } else {
      LLVM_DEBUG(spaces(indent);
                 llvm::dbgs() << "Ignored " << name << " because it does not have a type");
    }
  }

  return success();
}

static LogicalResult specializeTypeBindingImpl(
    TypeBinding *dst, ParamsScopeStack &scopes, const llvm::StringSet<> &FV,
    const TypeBindings &bindings, size_t indent
) {
  LLVM_DEBUG(spaces(indent); scopes.print(llvm::dbgs()); spaces(indent);
             printFVs(FV, llvm::dbgs()));
  // If the destionation is null do nothing
  if (dst == nullptr) {
    return success();
  }
  // If the type binding is a generic param replace it with the actual type.
  if (dst->isGenericParam()) {
    return specializeTypeBinding_genericParamCase(dst, scopes, FV, bindings, indent);
  }
  // If the type is a generic type apply the replacement to its generic parameters
  if (dst->isGeneric()) {
    if (failed(specializeTypeBinding_genericTypeCase(dst, scopes, FV, bindings, indent))) {
      return failure();
    }
    ParamsStorage sto;
    Params consts(sto);
    ParamsScopeStack constsScope(consts);
    return propagateConstants(dst, constsScope, bindings, indent);
  }

  // Do nothing in the other cases.
  LLVM_DEBUG(spaces(indent); dst->print(llvm::dbgs() << "Ignoring: ", true); llvm::dbgs() << "\n");
  return success();
}

#ifndef NDEBUG
static constexpr StringRef dividerLine =
    "//===------------------------------------------------------------------------===//";

#endif

namespace {
struct WrapLog {
  WrapLog(const TypeBinding &b) : binding(&b) {
    LLVM_DEBUG(llvm::dbgs() << dividerLine << "\n" << "Specializing " << b << "\n";);
  }
  ~WrapLog() {
    LLVM_DEBUG(llvm::dbgs() << "Finished specializing " << *binding << "\n"
                            << dividerLine << "\n";);
  }

private:
  const TypeBinding *binding;
};
} // namespace

mlir::LogicalResult zhl::specializeTypeBinding(
    TypeBinding *dst, ParamsScopeStack &scopes, const llvm::StringSet<> &FV,
    const TypeBindings &bindings
) {
  WrapLog w(*dst);
  return specializeTypeBindingImpl(dst, scopes, FV, bindings, 0);
}

mlir::LogicalResult zhl::specializeTypeBinding(
    TypeBinding *dst, ParamsScopeStack &scopes, const TypeBindings &bindings
) {
  WrapLog w(*dst);
  llvm::StringSet<> emptyFVs;
  return specializeTypeBindingImpl(dst, scopes, emptyFVs, bindings, 0);
}

mlir::FailureOr<zhl::TypeBinding> zhl::TypeBinding::specialize(
    EmitErrorFn emitError, mlir::ArrayRef<TypeBinding> params, TypeBindings &bindings
) const {
  if (specialized) {
    return emitError() << "can't respecialize type '" << getName() << "'";
  }
  auto declaredParamsSize = getGenericParamsMapping().sizeOfDeclared();
  if (declaredParamsSize == 0) {
    return emitError() << "type '" << name << "' is not generic";
  }
  if (declaredParamsSize != params.size()) {
    return emitError() << "wrong number of specialization parameters. Expected "
                       << declaredParamsSize << " but got " << params.size();
  }

  // The root scope for specialization is a mapping between the n-th generic
  // parameter and the n-th TypeBinding passed as argument to this method.
  ParamsMap generics;
  // Any type variable introduced by the specialization is a free variable.
  llvm::StringSet<> freeVariables;
  for (unsigned i = 0; i < params.size(); i++) {
    generics.declare(getGenericParamsMapping().getName(i), params[i], i);
    if (params[i].isGenericParam()) {
      freeVariables.insert(params[i].getGenericParamName());
    }
  }
  // Convert the lifted parameters to their expressions, contained in their super type
  auto genericParamsMapping = getGenericParamsMapping();
  size_t totalSize = genericParamsMapping.size();
  for (size_t i = params.size(); i < totalSize; i++) {
    assert(genericParamsMapping.getParam(i).hasSuperType());
    generics.declare(
        genericParamsMapping.getName(i), genericParamsMapping.getParam(i).getSuperType(), i
    );
  }

  TypeBinding specializedBinding(*this); // Make a copy to create the specialization

  ParamsStorage sto(generics);
  Params initialScope(sto);
  ParamsScopeStack scopeStack(initialScope);
  WrapLog w(*this);
  auto result =
      specializeTypeBindingImpl(&specializedBinding, scopeStack, freeVariables, bindings, 0);
  if (failed(result)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to specialize binding " << *this << "\n");
    return failure();
  }

  specializedBinding.markAsSpecialized();
  return specializedBinding;
}

#include "TypeBindings.h"

using namespace zhl;

Params::Params(ParamsMap map) : params(ParamsList(map.size())), names(ParamNames(map.size())) {
  for (auto &[k, type] : map) {
    params[k.second] = type;
    names[k.second] = k.first;
  }
}

Params::Params() = default;

Params::operator ParamsMap() const {
  ParamsMap map;
  for (size_t i = 0; i < params.size(); i++) {
    map[{names[i], i}] = params.at(i);
  }
  return map;
}

void Params::printNames(llvm::raw_ostream &os, char header, char footer) const {
  print<std::string>(names, os, [&](const auto &e) { os << e; }, header, footer);
}

void Params::printParams(llvm::raw_ostream &os, char header, char footer) const {
  print<TypeBinding>(params, os, [&](const auto &e) { e.print(os); }, header, footer);
}

TypeBinding Params::getParam(size_t i) const {
  assert(i < params.size());
  return params[i];
}

TypeBinding::TypeBinding(uint64_t value, const TypeBindings &bindings)
    : name(CONST), constVal(value), superType(&bindings.Get("Val")) {}

llvm::raw_ostream &zhl::operator<<(llvm::raw_ostream &os, const TypeBinding &type) {
  type.print(os);
  return os;
}
mlir::LogicalResult zhl::TypeBinding::subtypeOf(const TypeBinding &other) const {
  /*llvm::dbgs() << *this << " is a subtype of " << other << "? ";*/
  if (name == BOTTOM) {
    /*llvm::dbgs() << "Yes because is the bottom type\n";*/
    return mlir::success();
  }
  // TODO: Proper equality function
  if (getName() == other.getName()) {
    /*llvm::dbgs() << "Yes because they have the same name\n";*/
    return mlir::success();
  }

  if (superType != nullptr) {
    /*llvm::dbgs() << "Depends on the super type\n";*/
    return superType->subtypeOf(other);
  }
  /*llvm::dbgs() << "No\n";*/
  return mlir::failure();
}
zhl::TypeBinding::TypeBinding(
    llvm::StringRef name, const TypeBinding &superType, ParamsMap t_genericParams,
    ParamsMap t_constructorParams, MembersMap members
)
    : name(name), superType(&superType), members(members), genericParams(t_genericParams),
      constructorParams(t_constructorParams) {}

zhl::TypeBinding::TypeBinding(
    llvm::StringRef name, const TypeBinding &superType, ParamsMap t_genericParams
)
    : TypeBinding(name, superType, t_genericParams, {}, {}) {}

zhl::TypeBinding::TypeBinding(llvm::StringRef name, const TypeBinding &superType)
    : TypeBinding(name, superType, {}, {}, {}) {}

zhl::TypeBinding::TypeBinding() : name("Component"), superType(nullptr) {}
zhl::TypeBinding zhl::TypeBinding::commonSupertypeWith(const TypeBinding &other) const {
  if (mlir::succeeded(subtypeOf(other))) {
    return other;
  }
  if (mlir::succeeded(other.subtypeOf(*this))) {
    return *this;
  }

  // This algorithm is simple but is O(n^2) over the lengths of the subtyping chains.
  const TypeBinding *type = superType;
  while (type != nullptr && failed(other.subtypeOf(*type))) {
    type = type->superType;
  }
  return type != nullptr ? *type : TypeBinding();
}
mlir::FailureOr<TypeBinding>
zhl::TypeBinding::getArrayElement(std::function<mlir::InFlightDiagnostic()> emitError) const {
  if (!isArray()) {
    return emitError() << "non array type '" << name << "' cannot be subscripted";
  }
  assert(genericParams.size() == 2);
  if (!specialized) {
    return emitError() << "array type has not been specialized";
  }
  return genericParams.getParam(0);
}

mlir::FailureOr<TypeBinding> zhl::TypeBinding::specialize(
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
                       << genericParams.size() << " and got " << params.size();
  }
  ParamsMap generics;
  for (unsigned i = 0; i < params.size(); i++) {
    // TODO: Validation
    generics[{genericParams.getName(i), i}] = params[i];
  }

  TypeBinding specializedBinding(name, *superType, generics, constructorParams, members);
  specializedBinding.specialized = true;
  return variadic ? WrapVariadic(specializedBinding) : specializedBinding;
}

zhl::TypeBinding zhl::TypeBinding::WrapVariadic(const TypeBinding &t) {
  TypeBinding w = t;
  w.variadic = true;
  return w;
}
void zhl::TypeBinding::print(llvm::raw_ostream &os) const {
  if (isConst()) {
    if (constVal.has_value()) {
      os << *constVal;
    } else {
      os << "?";
    }
    return;
  }
  os << name;
  if (specialized) {
    genericParams.printParams(os);
  } else {
    genericParams.printNames(os);
  }
  if (variadic) {
    os << "...";
  }
}

std::string_view zhl::TypeBinding::getName() const { return name; }

std::string_view zhl::Params::getName(size_t i) const {
  assert(i < names.size());
  return names[i];
}

size_t zhl::Params::size() const { return params.size(); }

bool zhl::TypeBinding::isBottom() const { return name == BOTTOM; }

bool zhl::TypeBinding::isTypeMarker() const { return name == "Type"; }

bool zhl::TypeBinding::isVal() const { return name == "Val"; }

bool zhl::TypeBinding::isArray() const { return name == "Array"; }

bool zhl::TypeBinding::isConst() const { return name == CONST; }

const zhl::TypeBinding &zhl::TypeBindings::Component() { return bindings["Component"]; }

const zhl::TypeBinding &zhl::TypeBindings::Component() const { return bindings.at("Component"); }

const zhl::TypeBinding &zhl::TypeBindings::Bottom() const { return bottom; }

zhl::TypeBinding zhl::TypeBindings::Const(uint64_t value) const {
  return TypeBinding(value, *this);
}

zhl::TypeBinding zhl::TypeBindings::UnkConst() const { return TypeBinding(CONST, Get("Val")); }

zhl::TypeBinding zhl::TypeBindings::Array(TypeBinding type, uint64_t size) const {
  zhl::ParamsMap arrayGenericParams;
  arrayGenericParams[{"T", 0}] = type;
  arrayGenericParams[{"N", 1}] = Const(size);
  TypeBinding array("Array", Component(), arrayGenericParams);
  array.specialized = true;
  return array;
}

zhl::TypeBinding zhl::TypeBindings::UnkArray(TypeBinding type) const {
  zhl::ParamsMap arrayGenericParams;
  arrayGenericParams[{"T", 0}] = type;
  arrayGenericParams[{"N", 1}] = UnkConst();
  TypeBinding array("Array", Component(), arrayGenericParams);
  array.specialized = true;
  return array;
}

bool zhl::TypeBindings::Exists(std::string_view name) const {
  return bindings.find(name) != bindings.end();
}

const zhl::TypeBinding &zhl::TypeBindings::Get(std::string_view name) const {
  return bindings.at(name);
}

mlir::FailureOr<TypeBinding> zhl::TypeBindings::MaybeGet(std::string_view name) const {
  if (Exists(name)) {
    return Get(name);
  }
  return mlir::failure();
}

const zhl::TypeBinding &zhl::TypeBindings::Manage(const zhl::TypeBinding &binding) {
  managedBindings.push_back(binding);
  return managedBindings.back();
}

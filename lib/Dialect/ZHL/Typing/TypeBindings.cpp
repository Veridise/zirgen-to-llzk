#include "zklang/Dialect/ZHL/Typing/TypeBindings.h"
#include <iterator>
#include <mlir/Support/LogicalResult.h>

using namespace zhl;
using namespace mlir;

Params::Params(ParamsMap map) : names(ParamNames(map.size())) {
  // Hack to get the bindings ordered without having a default constructor
  std::vector<TypeBinding *> tmp(map.size());
  for (auto &[k, type] : map) {
    tmp[k.second] = &type;
    names[k.second] = k.first;
  }
  for (auto *type : tmp) {
    params.push_back(*type);
  }
}

Params::Params() = default;

Params::operator ParamsMap() const {
  ParamsMap map;
  for (size_t i = 0; i < params.size(); i++) {
    map.insert({{names[i], i}, params.at(i)});
  }
  return map;
}

void Params::printNames(llvm::raw_ostream &os, char header, char footer) const {
  print<std::string>(names, os, [&](const auto &e) { os << e; }, header, footer);
}

void Params::printParams(llvm::raw_ostream &os, char header, char footer) const {
  print<TypeBinding>(params, os, [&](const auto &e) { e.print(os); }, header, footer);
}

template <typename Elt>
void Params::print(
    const std::vector<Elt> &lst, llvm::raw_ostream &os, std::function<void(const Elt &)> handler,
    char header, char footer
) const {
  if (params.size() == 0) {
    return; // Don't print anything if there aren't any parameters
  }

  os << header;
  size_t c = 1;
  for (auto &e : lst) {
    handler(e);
    if (c < lst.size()) {
      os << ",";
    }
    c++;
  }
  os << footer;
}

TypeBinding Params::getParam(size_t i) const {
  assert(i < params.size());
  return params[i];
}

ArrayRef<std::string> Params::getNames() const { return names; }

Params::ParamsList::iterator Params::begin() { return params.begin(); }
Params::ParamsList::const_iterator Params::begin() const { return params.begin(); }
Params::ParamsList::iterator Params::end() { return params.end(); }
Params::ParamsList::const_iterator Params::end() const { return params.end(); }

TypeBinding::TypeBinding(
    uint64_t value, mlir::Location loc, const TypeBindings &bindings, bool isBuiltin
)
    : builtin(isBuiltin), name(CONST), loc(loc), constVal(value),
      superType(&const_cast<TypeBinding &>(bindings.Get("Val"))) {}

TypeBinding TypeBinding::WithUpdatedLocation(mlir::Location loc) const {
  TypeBinding b = *this;
  b.loc = loc;
  return b;
}

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

inline bool isPrivate(StringRef name) { return name.starts_with("_"); }

FailureOr<TypeBinding>
TypeBinding::getMember(StringRef name, std::function<mlir::InFlightDiagnostic()> emitError) const {
  if (members.find(name) == members.end()) {
    return emitError() << "member " << getName() << "." << name << " was not found";
  }
  if (isPrivate(name)) {
    return emitError() << "member " << getName() << "." << name
                       << " is private and cannot be accessed";
  }
  auto member = members.at(name);
  if (!member.has_value()) {
    return emitError() << "internal error: could not deduce the type of member " << getName() << "."
                       << name;
  }
  return *member;
}

zhl::TypeBinding::TypeBinding(
    llvm::StringRef name, mlir::Location loc, const TypeBinding &superType,
    ParamsMap t_genericParams, ParamsMap t_constructorParams, MembersMap members, bool isBuiltin
)
    : builtin(isBuiltin), name(name), loc(loc), superType(&const_cast<TypeBinding &>(superType)),
      members(members), genericParams(t_genericParams), constructorParams(t_constructorParams) {}

zhl::TypeBinding::TypeBinding(
    llvm::StringRef name, mlir::Location loc, const TypeBinding &superType,
    ParamsMap t_genericParams, bool isBuiltin
)
    : TypeBinding(name, loc, superType, t_genericParams, {}, {}, isBuiltin) {}

zhl::TypeBinding::TypeBinding(
    llvm::StringRef name, mlir::Location loc, const TypeBinding &superType, bool isBuiltin
)
    : TypeBinding(name, loc, superType, {}, {}, {}, isBuiltin) {}

zhl::TypeBinding::TypeBinding(mlir::Location loc)
    : builtin(true), name("Component"), loc(loc), superType(nullptr) {}
zhl::TypeBinding zhl::TypeBinding::commonSupertypeWith(const TypeBinding &other) const {
  if (mlir::succeeded(subtypeOf(other))) {
    print(llvm::dbgs());
    llvm::dbgs() << " is a sub type of ";
    other.print(llvm::dbgs());
    llvm::dbgs() << " and thus we return ";
    other.print(llvm::dbgs());
    llvm::dbgs() << "\n";
    return other;
  }
  if (mlir::succeeded(other.subtypeOf(*this))) {
    other.print(llvm::dbgs());
    llvm::dbgs() << " is a sub type of ";
    print(llvm::dbgs());
    llvm::dbgs() << " and thus we return ";
    print(llvm::dbgs());
    llvm::dbgs() << "\n";
    return *this;
  }

  // This algorithm is simple but is O(n^2) over the lengths of the subtyping chains.
  const TypeBinding *type = superType;
  while (type != nullptr && failed(other.subtypeOf(*type))) {
    type = type->superType;
  }
  return type != nullptr ? *type : TypeBinding(loc);
}
mlir::FailureOr<TypeBinding>
zhl::TypeBinding::getArrayElement(std::function<mlir::InFlightDiagnostic()> emitError) const {
  if (!isArray()) {
    return emitError() << "non array type '" << name << "' cannot be subscripted";
  }
  // A component with an array super type can behave like an array but it doesn't have all the
  // information required. In that case we defer the answer to the super type.
  if (name != "Array") {
    if (!hasSuperType()) {
      return failure();
    }
    return getSuperType().getArrayElement(emitError);
  }
  assert(genericParams.size() == 2);
  /*if (!specialized) {*/
  /*  return emitError() << "array type has not been specialized";*/
  /*}*/
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
    generics.insert({{genericParams.getName(i), i}, params[i]});
  }

  TypeBinding specializedBinding(name, loc, *superType, generics, constructorParams, members);
  specializedBinding.specialized = true;
  return variadic ? WrapVariadic(specializedBinding) : specializedBinding;
}

zhl::TypeBinding zhl::TypeBinding::WrapVariadic(const TypeBinding &t) {
  TypeBinding w = t;
  w.variadic = true;
  return w;
}

std::vector<mlir::Location> TypeBinding::getConstructorParamLocations() const {
  std::vector<mlir::Location> locs;
  std::transform(
      constructorParams.begin(), constructorParams.end(), std::back_inserter(locs),
      [](auto &binding) { return binding.loc; }
  );
  return locs;
}

const Params &TypeBinding::getConstructorParams() const { return constructorParams; }

void zhl::TypeBinding::print(llvm::raw_ostream &os) const {
  if (isConst()) {
    if (constVal.has_value()) {
      os << *constVal;
    } else {
      os << "?";
    }
    return;
  }
  if (isGenericParam()) {
    os << *genericParamName;
    return;
  }
  os << name;
  if (specialized) {
    genericParams.printParams(os);
  } else {
    genericParams.printNames(os);
  }
  constructorParams.printParams(os, '(', ')');
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

bool zhl::TypeBinding::isArray() const {
  return name == "Array" || (hasSuperType() && getSuperType().isArray());
}

bool zhl::TypeBinding::isBuiltin() const { return builtin; }

bool zhl::TypeBinding::isConst() const { return name == CONST; }

bool TypeBinding::isGeneric() const { return genericParams.size() > 0; }

bool TypeBinding::isSpecialized() const { return !isGeneric() || specialized; }

ArrayRef<std::string> TypeBinding::getGenericParamNames() const { return genericParams.getNames(); }

const zhl::TypeBinding &zhl::TypeBindings::Component() {
  if (!Exists("Component")) {
    bindings.insert({"Component", TypeBinding(unk)});
  }
  return bindings.at("Component");
}

const zhl::TypeBinding &zhl::TypeBindings::Component() const { return bindings.at("Component"); }

const zhl::TypeBinding &zhl::TypeBindings::Bottom() const { return bottom; }

zhl::TypeBinding zhl::TypeBindings::Const(uint64_t value) const { return Const(value, unk); }

zhl::TypeBinding zhl::TypeBindings::Const(uint64_t value, mlir::Location loc) const {
  return TypeBinding(value, loc, *this);
}

zhl::TypeBinding zhl::TypeBindings::UnkConst() const { return UnkConst(unk); }

zhl::TypeBinding zhl::TypeBindings::UnkConst(mlir::Location loc) const {
  return TypeBinding(CONST, loc, Get("Val"));
}

zhl::TypeBinding zhl::TypeBindings::Array(TypeBinding type, uint64_t size) const {
  return Array(type, size, unk);
}

zhl::TypeBinding
zhl::TypeBindings::Array(TypeBinding type, uint64_t size, mlir::Location loc) const {
  zhl::ParamsMap arrayGenericParams;
  arrayGenericParams.insert({{"T", 0}, type});
  arrayGenericParams.insert({{"N", 1}, Const(size)});
  TypeBinding array("Array", loc, Component(), arrayGenericParams, true);
  array.specialized = true;
  return array;
}

zhl::TypeBinding zhl::TypeBindings::UnkArray(TypeBinding type) const { return UnkArray(type, unk); }

zhl::TypeBinding zhl::TypeBindings::UnkArray(TypeBinding type, mlir::Location loc) const {
  zhl::ParamsMap arrayGenericParams;
  arrayGenericParams.insert({{"T", 0}, type});
  arrayGenericParams.insert({{"N", 1}, UnkConst()});
  TypeBinding array("Array", loc, Component(), arrayGenericParams, true);
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
  llvm::dbgs() << "No binding found for " << name << "\n";
  return mlir::failure();
}

const zhl::TypeBinding &zhl::TypeBindings::Manage(const zhl::TypeBinding &binding) {
  managedBindings.push_back(binding);
  return managedBindings.back();
}

TypeBindings::TypeBindings(OpBuilder &builder)
    : unk(builder.getUnknownLoc()), bottom(TypeBinding(BOTTOM, unk, Component())) {
  (void)Component();
}
void zhl::TypeBinding::selfConstructs() {
  if (selfConstructor) {
    return;
  }
  selfConstructor = true;
  ParamsMap map = constructorParams;
  map.clear();
  map.insert({{"x", 0}, *this});

  constructorParams = map;
}
const zhl::MembersMap &zhl::TypeBinding::getMembers() const { return members; }
mlir::Location zhl::TypeBinding::getLocation() const { return loc; }
const zhl::TypeBinding &zhl::TypeBinding::getSuperType() const {
  assert(superType != nullptr);
  return *superType;
}
bool zhl::TypeBinding::isVariadic() const { return variadic; }
mlir::ArrayRef<TypeBinding> zhl::Params::getParams() const { return params; }

mlir::ArrayRef<TypeBinding> zhl::TypeBinding::getGenericParams() const {
  return genericParams.getParams();
}
uint64_t zhl::TypeBinding::getConst() const {
  assert(constVal.has_value());
  return *constVal;
}
bool zhl::TypeBinding::hasSuperType() const { return superType != nullptr; }
bool zhl::TypeBinding::isGenericParam() const {
  if (superType == nullptr) {
    return false;
  }
  return genericParamName.has_value() && (superType->isTypeMarker() || superType->isVal());
}

zhl::TypeBinding zhl::TypeBinding::MakeGenericParam(const TypeBinding &t, llvm::StringRef name) {
  TypeBinding copy(name, t.loc, t);
  copy.genericParamName = name;
  return copy;
}
llvm::StringRef zhl::TypeBinding::getGenericParamName() const {
  assert(isGenericParam());
  return *genericParamName;
}
const zhl::Params &zhl::TypeBinding::getGenericParamsMapping() const { return genericParams; }
const TypeBinding *zhl::Params::operator[](std::string_view name) const {
  for (size_t i = 0; i < names.size(); i++) {
    if (names.at(i) == name) {
      return &params.at(i);
    }
  }
  return nullptr;
}

TypeBinding *zhl::Params::operator[](std::string_view name) {
  for (size_t i = 0; i < names.size(); i++) {
    if (names.at(i) == name) {
      return &params.at(i);
    }
  }
  return nullptr;
}
bool zhl::Params::empty() const { return names.empty(); }
void zhl::Params::replaceParam(std::string_view name, const TypeBinding &binding) {
  auto found = this->operator[](name);
  if (found != nullptr) {
    *found = binding;
  }
}
void zhl::TypeBinding::replaceGenericParamByName(
    std::string_view name, const TypeBinding &binding
) {
  genericParams.replaceParam(name, binding);
}
zhl::TypeBinding &zhl::TypeBinding::getSuperType() {
  assert(superType != nullptr);
  return *superType;
}

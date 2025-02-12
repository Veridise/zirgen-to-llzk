#include "zklang/Dialect/ZHL/Typing/TypeBindings.h"
#include <cassert>
#include <iterator>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>

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

void Params::printMapping(llvm::raw_ostream &os) const {
  os << "{ ";
  size_t c = 1;
  size_t siz = params.size();
  for (size_t i = 0; i < siz; i++) {
    os << names[i] << ": " << params.at(i);
    if (c < siz) {
      os << ", ";
    }
    c++;
  }
  os << " }";
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

TypeBinding TypeBinding::WithUpdatedLocation(mlir::Location newLoc) const {
  TypeBinding b = *this;
  b.loc = newLoc;
  return b;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TypeBinding &type) {
  type.print(os);
  return os;
}

mlir::Diagnostic &zhl::operator<<(mlir::Diagnostic &diag, const zhl::TypeBinding &b) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  b.print(ss);
  diag << s;
  return diag;
}

mlir::LogicalResult zhl::TypeBinding::subtypeOf(const TypeBinding &other) const {
  if (name == BOTTOM) {
    return mlir::success();
  }
  if (*this == other) {
    return mlir::success();
  }

  if (superType != nullptr) {
    return superType->subtypeOf(other);
  }
  return mlir::failure();
}

inline bool isPrivate(StringRef name) { return name.starts_with("_"); }

/// Locates the member in the inheritance chain. A component lower in the chain will shadow members
/// in components higher in the chain.
/// Returns failure if it was not found.
/// If it was found but it couldn't get typechecked returns success wrapping a nullopt.
FailureOr<std::optional<TypeBinding>> TypeBinding::locateMember(StringRef memberName) const {
  if (members.find(memberName) != members.end()) {
    return members.at(memberName);
  }
  if (superType != nullptr) {
    return superType->locateMember(memberName);
  }
  return failure();
}

FailureOr<TypeBinding> TypeBinding::getMember(
    StringRef memberName, std::function<mlir::InFlightDiagnostic()> emitError
) const {
  auto memberLookupRes = locateMember(memberName);
  if (failed(memberLookupRes)) {
    return emitError() << "member " << getName() << "." << memberName << " was not found";
  }
  // We check this after ensuring the member exists to give a more accurate error message
  if (isPrivate(memberName)) {
    return emitError() << "member " << getName() << "." << memberName
                       << " is private and cannot be accessed";
  }
  if (!memberLookupRes->has_value()) {
    return emitError() << "internal error: could not deduce the type of member " << getName() << "."
                       << memberName;
  }
  return **memberLookupRes;
}

zhl::TypeBinding::TypeBinding(
    llvm::StringRef name, mlir::Location loc, const TypeBinding &superType,
    ParamsMap t_genericParams, ParamsMap t_constructorParams, MembersMap members, Frame t_frame,
    bool isBuiltin
)
    : builtin(isBuiltin), name(name), loc(loc), superType(&const_cast<TypeBinding &>(superType)),
      members(members), genericParams(t_genericParams), constructorParams(t_constructorParams),
      frame(t_frame) {}

zhl::TypeBinding::TypeBinding(
    llvm::StringRef name, mlir::Location loc, const TypeBinding &superType,
    ParamsMap t_genericParams, Frame t_frame, bool isBuiltin
)
    : TypeBinding(name, loc, superType, t_genericParams, {}, {}, t_frame, isBuiltin) {}

zhl::TypeBinding::TypeBinding(
    llvm::StringRef name, mlir::Location loc, const TypeBinding &superType, Frame t_frame,
    bool isBuiltin
)
    : TypeBinding(name, loc, superType, {}, {}, {}, t_frame, isBuiltin) {}

zhl::TypeBinding::TypeBinding(mlir::Location loc)
    : builtin(true), name("Component"), loc(loc), superType(nullptr) {}

zhl::TypeBinding::TypeBinding(const TypeBinding &other)
    : variadic(other.variadic), specialized(other.specialized),
      selfConstructor(other.selfConstructor), builtin(other.builtin), name(other.name),
      loc(other.loc), constVal(other.constVal), genericParamName(other.genericParamName),
      superType(other.superType), members(other.members), genericParams(other.genericParams),
      constructorParams(other.constructorParams), frame(other.frame), slot(other.slot) {}

zhl::TypeBinding::TypeBinding(TypeBinding &&other)
    : variadic(std::move(other.variadic)), specialized(std::move(other.specialized)),
      selfConstructor(std::move(other.selfConstructor)), builtin(std::move(other.builtin)),
      name(std::move(other.name)), loc(std::move(other.loc)), constVal(std::move(other.constVal)),
      genericParamName(std::move(other.genericParamName)), superType(std::move(other.superType)),
      members(std::move(other.members)), genericParams(std::move(other.genericParams)),
      constructorParams(std::move(other.constructorParams)), frame(other.frame),
      slot(std::move(other.slot)) {}

zhl::TypeBinding &zhl::TypeBinding::operator=(const TypeBinding &other) {
  variadic = other.variadic;
  specialized = other.specialized;
  selfConstructor = other.selfConstructor;
  builtin = other.builtin;
  name = other.name;
  loc = other.loc;
  constVal = other.constVal;
  genericParamName = other.genericParamName;
  superType = other.superType;
  members = other.members;
  genericParams = other.genericParams;
  constructorParams = other.constructorParams;
  frame = other.frame;
  slot = other.slot;
  return *this;
}

zhl::TypeBinding &zhl::TypeBinding::operator=(TypeBinding &&other) {
  if (this != &other) {
    variadic = std::move(other.variadic);
    specialized = std::move(other.specialized);
    selfConstructor = std::move(other.selfConstructor);
    builtin = std::move(other.builtin);
    name = std::move(other.name);
    loc = std::move(other.loc);
    constVal = std::move(other.constVal);
    genericParamName = std::move(other.genericParamName);
    superType = std::move(other.superType);
    members = std::move(other.members);
    genericParams = std::move(other.genericParams);
    constructorParams = std::move(other.constructorParams);
    frame = other.frame;
    slot = std::move(other.slot);
  }
  return *this;
}

zhl::TypeBinding zhl::TypeBinding::commonSupertypeWith(const TypeBinding &other) const {
  if (mlir::succeeded(subtypeOf(other))) {
    print(llvm::dbgs());
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
  return genericParams.getParam(0);
}

FailureOr<TypeBinding> TypeBinding::getConcreteArrayType() const {
  if (!isArray()) {
    return failure();
  }
  if (name == "Array") {
    return *this;
  }
  return superType->getConcreteArrayType();
}

zhl::TypeBinding zhl::TypeBinding::WrapVariadic(const TypeBinding &t) {
  TypeBinding w = t;
  w.variadic = true;
  return w;
}

TypeBinding TypeBinding::ReplaceFrame(Frame newFrame) const {
  auto copy = *this;
  copy.frame = newFrame;
  return copy;
}

std::vector<mlir::Location> TypeBinding::getConstructorParamLocations() const {
  std::vector<mlir::Location> locs;
  std::transform(
      constructorParams.begin(), constructorParams.end(), std::back_inserter(locs),
      [](auto &binding) { return binding.loc; }
  );
  return locs;
}

bool Params::operator==(const Params &other) const {
  return params == other.params && names == other.names;
}

bool TypeBinding::operator==(const TypeBinding &other) const {
  bool superTypeIsEqual;
  if (superType == nullptr && other.superType == nullptr) {
    superTypeIsEqual = true;
  } else if (superType == nullptr || other.superType == nullptr) {
    superTypeIsEqual = false;
  } else {
    superTypeIsEqual = *superType == *other.superType;
  }

  return superTypeIsEqual && variadic == other.variadic && specialized == other.specialized &&
         selfConstructor == other.selfConstructor && builtin == other.builtin &&
         name == other.name && /*loc == other.loc &&*/ constVal == other.constVal &&
         genericParamName == other.genericParamName && members == other.members &&
         genericParams == other.genericParams && constructorParams == other.constructorParams;
}

const Params &TypeBinding::getConstructorParams() const { return constructorParams; }
Params &TypeBinding::getConstructorParams() { return constructorParams; }

void zhl::TypeBinding::print(llvm::raw_ostream &os, bool fullPrintout) const {
  if (isConst()) {
    if (constVal.has_value()) {
      os << *constVal;
    } else {
      os << "?";
    }
  } else if (isGenericParam()) {
    os << *genericParamName;
  } else {
    os << name;
    if (specialized) {
      genericParams.printParams(os);
    } else {
      genericParams.printNames(os);
    }
    if (fullPrintout) {
      constructorParams.printParams(os, '(', ')');
    }
    if (variadic) {
      os << "...";
    }
  }
  if (fullPrintout) {
    os << " { ";
    if (variadic) {
      os << "variadic ";
    }
    if (specialized) {
      os << "specialized ";
    }
    if (selfConstructor) {
      os << "selfConstructor ";
    }
    if (builtin) {
      os << "builtin ";
    }
    if (constVal.has_value()) {
      os << "const(" << *constVal << ") ";
    }
    if (genericParamName.has_value()) {
      os << "genericParam(" << *genericParamName << ") ";
    }
    os << "}";
    if (!members.empty()) {
      os << " members {";
      size_t c = 1;
      size_t siz = members.size();
      for (auto &[memberName, type] : members) {
        os << memberName << ": ";
        if (type.has_value()) {
          type->print(os);
        } else {
          os << "⊥";
        }
        if (c < siz) {
          os << ", ";
        }
        c++;
      }
      os << " }";
    }
    if (hasSuperType()) {
      os << " :> ";
      getSuperType().print(os, fullPrintout);
    }
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

bool zhl::TypeBinding::isKnownConst() const { return isConst() && constVal.has_value(); }

bool TypeBinding::isGeneric() const { return genericParams.size() > 0; }

bool TypeBinding::isSpecialized() const { return !isGeneric() || specialized; }

void TypeBinding::markAsSpecialized() {
  assert(isGeneric());
  specialized = true;
}

ArrayRef<std::string> TypeBinding::getGenericParamNames() const { return genericParams.getNames(); }

void TypeBinding::markSlot(FrameSlot *newSlot) { slot = newSlot; }

FrameSlot *TypeBinding::getSlot() const { return slot; }

Frame TypeBinding::getFrame() const { return frame; }

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
  TypeBinding array("Array", loc, Component(), arrayGenericParams, Frame(), true);
  array.specialized = true;
  return array;
}

zhl::TypeBinding zhl::TypeBindings::UnkArray(TypeBinding type) const { return UnkArray(type, unk); }

zhl::TypeBinding zhl::TypeBindings::UnkArray(TypeBinding type, mlir::Location loc) const {
  zhl::ParamsMap arrayGenericParams;
  arrayGenericParams.insert({{"T", 0}, type});
  arrayGenericParams.insert({{"N", 1}, UnkConst()});
  TypeBinding array("Array", loc, Component(), arrayGenericParams, Frame(), true);
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

TypeBindings::TypeBindings(Location defaultLoc)
    : unk(defaultLoc), bottom(TypeBinding(BOTTOM, unk, Component())) {
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
zhl::MembersMap &zhl::TypeBinding::getMembers() { return members; }
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
    std::string_view paramName, const TypeBinding &binding
) {
  genericParams.replaceParam(paramName, binding);
}
zhl::TypeBinding &zhl::TypeBinding::getSuperType() {
  assert(superType != nullptr);
  return *superType;
}

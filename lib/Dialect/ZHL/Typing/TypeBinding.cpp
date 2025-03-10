#include <cassert>
#include <iterator>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Expr.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/Params.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

#define DEBUG_TYPE "zhl-type-bindings"

using namespace zhl;
using namespace mlir;

namespace {

inline bool hasConstValue(const expr::ConstExpr &constExpr) {
  return constExpr && mlir::isa<expr::ValExpr>(constExpr);
}

inline uint64_t getConstValue(const expr::ConstExpr &constExpr) {
  return mlir::cast<expr::ValExpr>(constExpr).getValue();
}

} // namespace

TypeBinding::TypeBinding(
    uint64_t value, mlir::Location loc, const TypeBindings &bindings, bool isBuiltin
)
    : builtin(isBuiltin), name(CONST), loc(loc), constExpr(expr::ConstExpr::Val(value)),
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

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TypeBinding::Name &name) {
  os << name.ref();
  return os;
}

mlir::Diagnostic &zhl::operator<<(mlir::Diagnostic &diag, const TypeBinding &b) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  b.print(ss);
  diag << s;
  return diag;
}

mlir::Diagnostic &zhl::operator<<(mlir::Diagnostic &diag, const TypeBinding::Name &name) {
  diag << Twine(name.ref());
  return diag;
}

mlir::LogicalResult TypeBinding::subtypeOf(const TypeBinding &other) const {
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

TypeBinding::~TypeBinding() = default;

TypeBinding::TypeBinding(
    llvm::StringRef name, mlir::Location loc, const TypeBinding &superType,
    ParamsMap t_genericParams, ParamsMap t_constructorParams, MembersMap members, Frame t_frame,
    bool isBuiltin
)
    : builtin(isBuiltin), name(name), loc(loc), superType(&const_cast<TypeBinding &>(superType)),
      members(members), genericParams(t_genericParams), constructorParams(t_constructorParams),
      frame(t_frame) {}

TypeBinding::TypeBinding(
    llvm::StringRef name, mlir::Location loc, const TypeBinding &superType,
    ParamsMap t_genericParams, Frame t_frame, bool isBuiltin
)
    : TypeBinding(name, loc, superType, t_genericParams, {}, {}, t_frame, isBuiltin) {}

TypeBinding::TypeBinding(
    llvm::StringRef name, mlir::Location loc, const TypeBinding &superType, Frame t_frame,
    bool isBuiltin
)
    : TypeBinding(name, loc, superType, {}, {}, {}, t_frame, isBuiltin) {}

TypeBinding::TypeBinding(mlir::Location loc)
    : builtin(true), name("Component"), loc(loc), superType(nullptr) {}

TypeBinding::TypeBinding(const TypeBinding &other)
    : variadic(other.variadic), specialized(other.specialized),
      selfConstructor(other.selfConstructor), builtin(other.builtin), closure(other.closure),
      name(other.name), loc(other.loc), constExpr(other.constExpr),
      genericParamName(other.genericParamName), superType(other.superType), members(other.members),
      genericParams(other.genericParams), constructorParams(other.constructorParams),
      frame(other.frame), slot(other.slot) {}

TypeBinding::TypeBinding(TypeBinding &&other)
    : variadic(std::move(other.variadic)), specialized(std::move(other.specialized)),
      selfConstructor(std::move(other.selfConstructor)), builtin(std::move(other.builtin)),
      closure(std::move(other.closure)), name(std::move(other.name)), loc(std::move(other.loc)),
      constExpr(std::move(other.constExpr)), genericParamName(std::move(other.genericParamName)),
      superType(std::move(other.superType)), members(std::move(other.members)),
      genericParams(std::move(other.genericParams)),
      constructorParams(std::move(other.constructorParams)), frame(other.frame),
      slot(std::move(other.slot)) {}

TypeBinding &TypeBinding::operator=(const TypeBinding &other) {
  variadic = other.variadic;
  specialized = other.specialized;
  selfConstructor = other.selfConstructor;
  builtin = other.builtin;
  closure = other.closure;
  name = other.name;
  loc = other.loc;
  constExpr = other.constExpr;
  genericParamName = other.genericParamName;
  superType = other.superType;
  members = other.members;
  genericParams = other.genericParams;
  constructorParams = other.constructorParams;
  frame = other.frame;
  slot = other.slot;
  return *this;
}

TypeBinding &TypeBinding::operator=(TypeBinding &&other) {
  if (this != &other) {
    variadic = std::move(other.variadic);
    specialized = std::move(other.specialized);
    selfConstructor = std::move(other.selfConstructor);
    builtin = std::move(other.builtin);
    closure = std::move(other.closure);
    name = std::move(other.name);
    loc = std::move(other.loc);
    constExpr = std::move(other.constExpr);
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

TypeBinding TypeBinding::commonSupertypeWith(const TypeBinding &other) const {
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
  return type != nullptr ? *type : TypeBinding(loc);
}
mlir::FailureOr<TypeBinding> TypeBinding::getArrayElement(EmitErrorFn emitError) const {
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
  assert(getGenericParamsMapping().size() == 2);
  return getGenericParamsMapping().getParam(0);
}

mlir::FailureOr<TypeBinding> TypeBinding::getArraySize(EmitErrorFn emitError) const {
  if (!isArray()) {
    return emitError() << "non array type '" << name << "' cannot be subscripted";
  }
  // A component with an array super type can behave like an array but it doesn't have all the
  // information required. In that case we defer the answer to the super type.
  if (name != "Array") {
    if (!hasSuperType()) {
      return failure();
    }
    return getSuperType().getArraySize(emitError);
  }
  assert(getGenericParamsMapping().size() == 2);
  return getGenericParamsMapping().getParam(1);
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

TypeBinding TypeBinding::WrapVariadic(const TypeBinding &t) {
  TypeBinding w = t;
  w.variadic = true;
  return w;
}

TypeBinding TypeBinding::ReplaceFrame(Frame newFrame) const {
  auto copy = *this;
  copy.frame = newFrame;
  return copy;
}

SmallVector<Location> TypeBinding::getConstructorParamLocations() const {
  SmallVector<Location> locs;
  std::transform(
      getConstructorParams().begin(), getConstructorParams().end(), std::back_inserter(locs),
      [](auto &binding) { return binding.loc; }
  );
  return locs;
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
         name == other.name && constExpr == other.constExpr &&
         genericParamName == other.genericParamName && members == other.members &&
         getGenericParamsMapping() == other.getGenericParamsMapping() &&
         getConstructorParams() == other.getConstructorParams();
}

Params TypeBinding::getConstructorParams() const { return constructorParams; }
MutableParams TypeBinding::getConstructorParams() { return constructorParams; }

ParamsStorage *TypeBinding::ParamsStorageFactory::init() { return new ParamsStorage(); }

TypeBinding::ParamsStoragePtr::operator Params() const { return Params(this->operator*()); }
TypeBinding::ParamsStoragePtr::operator MutableParams() { return MutableParams(this->operator*()); }

TypeBinding::ParamsStoragePtr &TypeBinding::ParamsStoragePtr::operator=(ParamsMap &map) {
  set(new ParamsStorage(map));
  return *this;
}
TypeBinding::ParamsStoragePtr &TypeBinding::ParamsStoragePtr::operator=(ParamsMap &&map) {
  set(new ParamsStorage(map));
  return *this;
}

TypeBinding::ParamsStoragePtr::ParamsStoragePtr(ParamsMap &map)
    : zklang::CopyablePointer<ParamsStorage, ParamsStorageFactory>(new ParamsStorage(map)) {}

const TypeBinding &TypeBinding::StripConst(const TypeBinding &binding) {
  if (binding.isConst()) {
    return binding.getSuperType();
  }
  return binding;
}

TypeBinding TypeBinding::WithClosure(const TypeBinding &binding) {
  auto copy = binding;
  copy.closure = true;
  return copy;
}

TypeBinding TypeBinding::WithoutClosure(const TypeBinding &binding) {
  auto copy = binding;
  copy.closure = false;
  return copy;
}

bool TypeBinding::hasClosure() const { return closure; }

void TypeBinding::setName(StringRef newName) {
  // If the binding has a slot check if it can be converted to a ComponentSlot,
  // assert that the type is equal (before the change of name)
  // and then rename the inner binding in the slot.
  if (auto *compSlot = mlir::dyn_cast_if_present<ComponentSlot>(slot)) {
    assert(compSlot->contains(*this));
    compSlot->editInnerBinding([&](TypeBinding &inner) {
      if (this != &inner) {
        inner.setName(newName);
      }
    });
  }
  name = newName;
}

void TypeBinding::print(llvm::raw_ostream &os, bool fullPrintout) const {
  if (isConst()) {
    if (hasConstValue(constExpr)) {
      os << getConstValue(constExpr);
    } else {
      os << "?";
    }
  } else if (isGenericParam()) {
    os << *genericParamName;
  } else {
    os << name.ref();
    if (specialized) {
      getGenericParamsMapping().printParams(os, false);
    } else {
      getGenericParamsMapping().printNames(os);
    }
    if (fullPrintout) {
      getConstructorParams().printParams(os, false, '(', ')');
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
    if (closure) {
      os << "closure ";
    }
    if (hasConstValue(constExpr)) {
      os << "const(" << getConstValue(constExpr) << ") ";
    }
    if (genericParamName.has_value()) {
      os << "genericParam(" << *genericParamName << ") ";
    }
    os << "constExpr(";
    if (constExpr) {
      constExpr->print(os);
    } else {
      os << "<<NULL>>";
    }
    os << ") ";
    os << "}";
    if (!members.empty()) {
      os << " members { ";
      size_t c = 1;
      size_t siz = members.size();
      for (auto &[memberName, type] : members) {
        os << memberName << ": ";
        if (type.has_value()) {
          type->print(os, false);
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

mlir::StringRef TypeBinding::getName() const { return name.ref(); }

bool TypeBinding::isBottom() const { return name.ref() == BOTTOM; }

bool TypeBinding::isTypeMarker() const { return name.ref() == "Type"; }

static bool isTransitivelyValImpl(const TypeBinding &type) {
  const auto *t = &type;
  while (t->hasSuperType()) {
    t = &t->getSuperType();
    if (t->getName() == "Val") {
      return true;
    }
  }
  return false;
}

bool TypeBinding::isVal() const { return name.ref() == "Val"; }

bool TypeBinding::isTransitivelyVal() const {
  return hasConstExpr() && isTransitivelyValImpl(*this);
}

bool TypeBinding::isArray() const {
  return name == "Array" || (hasSuperType() && getSuperType().isArray());
}

bool TypeBinding::isBuiltin() const { return builtin; }

bool TypeBinding::isConst() const { return name.ref() == CONST; }

bool TypeBinding::isKnownConst() const { return isConst() && hasConstValue(constExpr); }

bool TypeBinding::isGeneric() const { return getGenericParamsMapping().size() > 0; }

bool TypeBinding::isSpecialized() const { return !isGeneric() || specialized; }

void TypeBinding::markAsSpecialized() {
  assert(isGeneric());
  specialized = true;
}

ArrayRef<ParamName> TypeBinding::getGenericParamNames() const {
  return getGenericParamsMapping().getNames();
}

#ifndef NDEBUG
static llvm::raw_ostream &p(FrameSlot *slot) {
  if (slot) {
    slot->print(llvm::dbgs() << slot << " ");
  } else {
    llvm::dbgs() << "<<NULL>>";
  }
  return llvm::dbgs();
}
#endif

void TypeBinding::markSlot(FrameSlot *newSlot) {
  if (slot == newSlot) {
    return;
  }
  LLVM_DEBUG(print(llvm::dbgs() << "Binding "); llvm::dbgs() << ": Current slot is ";
             p(slot) << " and the new slot is "; p(newSlot) << "\n");
  assert((!slot == !!newSlot) && "Writing over an existing slot!");
  slot = newSlot;
}

FrameSlot *TypeBinding::getSlot() const { return slot; }

Frame TypeBinding::getFrame() const { return frame; }

void TypeBinding::selfConstructs() {
  if (selfConstructor) {
    return;
  }
  selfConstructor = true;
  constructorParams = ParamsMap().declare("x", *this);
}

const MembersMap &TypeBinding::getMembers() const { return members; }
MembersMap &TypeBinding::getMembers() { return members; }
mlir::Location TypeBinding::getLocation() const { return loc; }
const TypeBinding &TypeBinding::getSuperType() const {
  assert(superType != nullptr);
  return *superType;
}
bool TypeBinding::isVariadic() const { return variadic; }

MutableArrayRef<TypeBinding> TypeBinding::getGenericParams() {
  return getGenericParamsMapping().getParams();
}

ArrayRef<TypeBinding> TypeBinding::getGenericParams() const {
  return getGenericParamsMapping().getParams();
}

SmallVector<TypeBinding, 0> TypeBinding::getDeclaredGenericParams() const {
  return getGenericParamsMapping().getDeclaredParams();
}

uint64_t TypeBinding::getConst() const {
  assert(hasConstValue(constExpr));
  return getConstValue(constExpr);
}
bool TypeBinding::hasSuperType() const { return superType != nullptr; }
bool TypeBinding::isGenericParam() const {
  if (superType == nullptr) {
    return false;
  }
  return genericParamName.has_value() &&
         (superType->isTypeMarker() || superType->isVal() || superType->isTransitivelyVal());
}

TypeBinding TypeBinding::MakeGenericParam(const TypeBinding &t, llvm::StringRef name) {
  TypeBinding copy(name, t.loc, t);
  copy.genericParamName = name;
  return copy;
}
llvm::StringRef TypeBinding::getGenericParamName() const {
  assert(isGenericParam());
  return *genericParamName;
}

Params TypeBinding::getGenericParamsMapping() const { return genericParams; }

MutableParams TypeBinding::getGenericParamsMapping() { return genericParams; }

void TypeBinding::replaceGenericParamByName(StringRef paramName, const TypeBinding &binding) {
  getGenericParamsMapping().replaceParam(paramName, binding);
}
TypeBinding &TypeBinding::getSuperType() {
  assert(superType != nullptr);
  return *superType;
}

TypeBinding TypeBinding::WithExpr(const TypeBinding &b, expr::ConstExpr constExpr) {
  auto copy = b;
  copy.constExpr = constExpr;
  return copy;
}

TypeBinding TypeBinding::NoExpr(const TypeBinding &b) {
  auto copy = b;
  copy.constExpr = expr::ConstExpr();
  return copy;
}

bool TypeBinding::hasConstExpr() const { return constExpr; }

const expr::ConstExpr &TypeBinding::getConstExpr() const { return constExpr; }

#include <cassert>
#include <iterator>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Expr.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
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

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TypeBindingName &name) {
  os << name.ref();
  return os;
}

mlir::Diagnostic &zhl::operator<<(mlir::Diagnostic &diag, const zhl::TypeBinding &b) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  b.print(ss);
  diag << s;
  return diag;
}

mlir::Diagnostic &zhl::operator<<(mlir::Diagnostic &diag, const zhl::TypeBindingName &name) {
  diag << Twine(name.ref());
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
      selfConstructor(other.selfConstructor), builtin(other.builtin), closure(other.closure),
      name(other.name), loc(other.loc), /*constVal(other.constVal),*/ constExpr(other.constExpr),
      genericParamName(other.genericParamName), superType(other.superType), members(other.members),
      genericParams(other.genericParams), constructorParams(other.constructorParams),
      frame(other.frame), slot(other.slot) {}

zhl::TypeBinding::TypeBinding(TypeBinding &&other)
    : variadic(std::move(other.variadic)), specialized(std::move(other.specialized)),
      selfConstructor(std::move(other.selfConstructor)), builtin(std::move(other.builtin)),
      closure(std::move(other.closure)), name(std::move(other.name)),
      loc(std::move(other.loc)), /*constVal(std::move(other.constVal)),*/
      constExpr(std::move(other.constExpr)), genericParamName(std::move(other.genericParamName)),
      superType(std::move(other.superType)), members(std::move(other.members)),
      genericParams(std::move(other.genericParams)),
      constructorParams(std::move(other.constructorParams)), frame(other.frame),
      slot(std::move(other.slot)) {}

zhl::TypeBinding &zhl::TypeBinding::operator=(const TypeBinding &other) {
  variadic = other.variadic;
  specialized = other.specialized;
  selfConstructor = other.selfConstructor;
  builtin = other.builtin;
  closure = other.closure;
  name = other.name;
  loc = other.loc;
  // constVal = other.constVal;
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

zhl::TypeBinding &zhl::TypeBinding::operator=(TypeBinding &&other) {
  if (this != &other) {
    variadic = std::move(other.variadic);
    specialized = std::move(other.specialized);
    selfConstructor = std::move(other.selfConstructor);
    builtin = std::move(other.builtin);
    closure = std::move(other.closure);
    name = std::move(other.name);
    loc = std::move(other.loc);
    // constVal = std::move(other.constVal);
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
mlir::FailureOr<TypeBinding> zhl::TypeBinding::getArrayElement(EmitErrorFn emitError) const {
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

mlir::FailureOr<TypeBinding> zhl::TypeBinding::getArraySize(EmitErrorFn emitError) const {
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
  assert(genericParams.size() == 2);
  return genericParams.getParam(1);
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
         name == other.name && /*loc == other.loc &&*/ /* constVal == other.constVal &&*/
                                   constExpr == other.constExpr &&
         genericParamName == other.genericParamName && members == other.members &&
         genericParams == other.genericParams && constructorParams == other.constructorParams;
}

const Params &TypeBinding::getConstructorParams() const { return constructorParams; }
Params &TypeBinding::getConstructorParams() { return constructorParams; }

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

void zhl::TypeBinding::print(llvm::raw_ostream &os, bool fullPrintout) const {
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
      genericParams.printParams(os, false);
    } else {
      genericParams.printNames(os);
    }
    if (fullPrintout) {
      constructorParams.printParams(os, false, '(', ')');
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

std::string_view zhl::TypeBinding::getName() const { return name.ref(); }

bool zhl::TypeBinding::isBottom() const { return name.ref() == BOTTOM; }

bool zhl::TypeBinding::isTypeMarker() const { return name.ref() == "Type"; }

bool zhl::TypeBinding::isVal() const { return name.ref() == "Val"; }

bool zhl::TypeBinding::isArray() const {
  return name == "Array" || (hasSuperType() && getSuperType().isArray());
}

bool zhl::TypeBinding::isBuiltin() const { return builtin; }

bool zhl::TypeBinding::isConst() const { return name.ref() == CONST; }

bool zhl::TypeBinding::isKnownConst() const {
  return isConst() && /*constVal.has_value()*/ hasConstValue(constExpr);
}

bool TypeBinding::isGeneric() const { return genericParams.size() > 0; }

bool TypeBinding::isSpecialized() const { return !isGeneric() || specialized; }

void TypeBinding::markAsSpecialized() {
  assert(isGeneric());
  specialized = true;
}

ArrayRef<std::string> TypeBinding::getGenericParamNames() const { return genericParams.getNames(); }

#ifndef NDEBUG
llvm::raw_ostream &p(FrameSlot *slot) {
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

mlir::ArrayRef<TypeBinding> zhl::TypeBinding::getGenericParams() const {
  return genericParams.getParams();
}
uint64_t zhl::TypeBinding::getConst() const {
  assert(hasConstValue(constExpr));
  return getConstValue(constExpr);
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

zhl::Params &zhl::TypeBinding::getGenericParamsMapping() { return genericParams; }

void zhl::TypeBinding::replaceGenericParamByName(
    std::string_view paramName, const TypeBinding &binding
) {
  genericParams.replaceParam(paramName, binding);
}
zhl::TypeBinding &zhl::TypeBinding::getSuperType() {
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

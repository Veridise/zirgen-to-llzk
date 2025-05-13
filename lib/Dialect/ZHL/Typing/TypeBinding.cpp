//===- TypeBinding.cpp - Type information & metadata ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Expr.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/Params.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

#define DEBUG_TYPE "zhl-type-bindings"

using namespace zhl;
using namespace mlir;

//==-----------------------------------------------------------------------==//
// Helper functions
//==-----------------------------------------------------------------------==//

static bool hasConstValue(const expr::ConstExpr constExpr) {
  return constExpr && mlir::isa<expr::ValExpr>(constExpr);
}

static uint64_t getConstValue(const expr::ConstExpr constExpr) {
  assert(hasConstValue(constExpr));
  return mlir::cast<expr::ValExpr>(constExpr)->getValue();
}

static bool isPrivate(StringRef name) { return name.starts_with("_"); }

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

// Debugging helpers
#ifndef NDEBUG
namespace {
struct deferDecr {
  ~deferDecr() { value--; }

  unsigned &value;
};
} // namespace

static llvm::raw_ostream &p(FrameSlot *slot) {
  if (slot) {
    slot->print(llvm::dbgs() << slot << " ");
  } else {
    llvm::dbgs() << "<<NULL>>";
  }
  return llvm::dbgs();
}
#endif

//==-----------------------------------------------------------------------==//
// TypeBinding
//==-----------------------------------------------------------------------==//

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

mlir::LogicalResult TypeBinding::subtypeOf(const TypeBinding &other) const {
  if (name == BOTTOM) {
    return mlir::success();
  }
  if (*this == other) {
    return mlir::success();
  }
  // Special case for arrays
  if (name == "Array" && other.name == "Array") {
    auto &arrElt = getGenericParamsMapping().getParam(0);
    auto &otherArrElt = other.getGenericParamsMapping().getParam(0);
    auto &arrSize = getGenericParamsMapping().getParam(1);
    auto &otherArrSize = other.getGenericParamsMapping().getParam(1);
    if (arrSize == otherArrSize) {
      return arrElt.subtypeOf(otherArrElt);
    }
    return failure();
  }

  if (superType != nullptr) {
    return superType->subtypeOf(other);
  }
  return mlir::failure();
}

TypeBinding TypeBinding::commonSupertypeWith(const TypeBinding &other) const {
// Wrapped this way instead of LLVM_DEBUG because the macro may create its own scope.
#ifndef NDEBUG
  static unsigned indentLevel = 0;
  indentLevel++;
  deferDecr defer{.value = indentLevel};
  auto logLine = [&](unsigned offset = 0) -> llvm::raw_ostream & {
    return llvm::dbgs().indent(indentLevel + offset);
  };
  auto nl = []() { llvm::dbgs() << '\n'; };
#endif
  LLVM_DEBUG(print(logLine() << " this  = ", true); nl();
             other.print(logLine() << " other = ", true); nl(););
  if (*this == other) {
    LLVM_DEBUG(logLine(1) << "They are equal\n");
    return *this;
  }
  // Special cases for arrays:
  if (name == "Array") {
    // If they are the same size the common super type is another array
    // whose element is the least common type of the two element types
    if (other.name == "Array") {
      auto &arrElt = getGenericParamsMapping().getParam(0);
      LLVM_DEBUG(arrElt.print(logLine(1) << "This array element:  ", true); nl());
      auto &otherArrElt = other.getGenericParamsMapping().getParam(0);
      LLVM_DEBUG(otherArrElt.print(logLine(1) << "Other array element: ", true); nl());
      auto &arrSize = getGenericParamsMapping().getParam(1);
      LLVM_DEBUG(arrSize.print(logLine(1) << "This array size:     ", true); nl());
      auto &otherArrSize = other.getGenericParamsMapping().getParam(1);
      LLVM_DEBUG(otherArrSize.print(logLine(1) << "Other array size:    ", true); nl());
      if (arrSize == otherArrSize) {
        auto commonInner = arrElt.commonSupertypeWith(otherArrElt);
        auto copy = *this;
        copy.getGenericParamsMapping().getParam(0) = commonInner;
        copy.selfConstructs();
        LLVM_DEBUG(copy.print(logLine(1) << "Resolved a new array: ", true); nl());
        return copy;
      }
      LLVM_DEBUG(logLine(1) << "Defaulting to Component\n");
      return TypeBinding(loc);
    }
    // If this is an Array and the other has an Array supertype get the Array super type and go from
    // there.
    auto otherArray = other.getConcreteArrayType();
    if (succeeded(otherArray)) {
      LLVM_DEBUG(otherArray->print(logLine(1) << "Jumped to Array super type: ", true); nl());
      return commonSupertypeWith(*otherArray);
    }
  }
  if (mlir::succeeded(subtypeOf(other))) {
    LLVM_DEBUG(logLine(1) << "this is a subtype of other\n");
    return other;
  }
  if (mlir::succeeded(other.subtypeOf(*this))) {
    LLVM_DEBUG(logLine(1) << "other is a subtype of this\n");
    return *this;
  }

  if (superType) {
    // Climb alternatively
    LLVM_DEBUG(superType->print(logLine(1) << "Delegating to super type: ", true); nl());
    return other.commonSupertypeWith(*superType);
  }
  LLVM_DEBUG(logLine(1) << "Defaulting to Component\n");
  return TypeBinding(loc);
}

bool TypeBinding::isTransitivelyVal() const {
  return hasConstExpr() && isTransitivelyValImpl(*this);
}

bool TypeBinding::isArray() const {
  return name == "Array" || (hasSuperType() && getSuperType().isArray());
}

bool TypeBinding::isKnownConst() const { return isConst() && hasConstValue(constExpr); }

bool TypeBinding::isGenericParam() const {
  if (superType == nullptr) {
    return false;
  }
  return flags.isGenericParamName() &&
         (superType->isTypeMarker() || superType->isVal() || superType->isTransitivelyVal());
}

void TypeBinding::markSlot(FrameSlot *newSlot) {
  if (slot == newSlot) {
    return;
  }
  LLVM_DEBUG(print(llvm::dbgs() << "Binding "); llvm::dbgs() << ": Current slot is ";
             p(slot) << " and the new slot is "; p(newSlot) << '\n');
  assert((!slot == !!newSlot) && "Writing over an existing slot!");
  slot = newSlot;
}

uint64_t TypeBinding::getConst() const {
  assert(hasConstValue(constExpr));
  return getConstValue(constExpr);
}

SmallVector<Location> TypeBinding::getConstructorParamLocations() const {
  return llvm::map_to_vector(getConstructorParams(), [](auto &binding) { return binding.loc; });
}

FailureOr<TypeBinding> TypeBinding::getMember(StringRef memberName, EmitErrorFn emitError) const {
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

FailureOr<TypeBinding> TypeBinding::map(Params params, EmitErrorFn emitError) const {
  if (isGenericParam() && params.contains(getGenericParamName())) {
    auto result = params[getGenericParamName()];
    if (!result) {
      return emitError() << "parameter '" << getGenericParamName() << "' not found";
    }
    return *result;
  }
  if (hasConstExpr()) {
    auto replacement = constExpr.remap(params, emitError);
    if (failed(replacement)) {
      return failure();
    }
    return WithExpr(*this, *replacement);
  }
  return *this;
}

static FailureOr<TypeBinding>
maybeMap(FailureOr<TypeBinding> binding, Params params, EmitErrorFn emitError) {
  if (failed(binding)) {
    return binding;
  }
  return binding->map(params, emitError);
}

static mlir::FailureOr<TypeBinding>
getArrayProperty(size_t idx, EmitErrorFn emitError, const TypeBinding &binding) {
  auto name = binding.getName();
  auto params = binding.getGenericParamsMapping();
  if (!binding.isArray()) {
    return emitError() << "non array type '" << name << "' cannot be subscripted";
  }
  if (name == "Array") {
    assert(params.size() == 2);
    return params.getParam(idx);
  }
  // A component with an array super type can behave like an array but it doesn't have all the
  // information required. In that case we defer the answer to the super type.
  if (!binding.hasSuperType()) {
    return failure();
  }
  // Get the property from the parent and adapt it to this context.
  return maybeMap(getArrayProperty(idx, emitError, binding.getSuperType()), params, emitError);
}

mlir::FailureOr<TypeBinding> TypeBinding::getArrayElement(EmitErrorFn emitError) const {
  return getArrayProperty(0, emitError, *this);
}

mlir::FailureOr<TypeBinding> TypeBinding::getArraySize(EmitErrorFn emitError) const {
  return getArrayProperty(1, emitError, *this);
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

TypeBinding::TypeBinding(mlir::Location Loc)
    : flags(Flags::MkBuiltin(true)), name("Component"), loc(Loc), superType(nullptr) {}

TypeBinding::TypeBinding(
    llvm::StringRef BindingName, mlir::Location Loc, const TypeBinding &SuperType, Frame Frame,
    bool isBuiltin
)
    : TypeBinding(BindingName, Loc, SuperType, {}, {}, {}, Flags::MkBuiltin(isBuiltin), Frame) {}

TypeBinding::TypeBinding(
    StringRef BindingName, mlir::Location Loc, const TypeBinding &SuperType, Flags BindingFlags,
    Frame Frame
)
    : TypeBinding(BindingName, Loc, SuperType, {}, {}, {}, BindingFlags, Frame) {}

TypeBinding::TypeBinding(
    llvm::StringRef BindingName, mlir::Location Loc, const TypeBinding &SuperType,
    ParamsMap GenericParams, Frame Frame, bool isBuiltin
)
    : TypeBinding(BindingName, Loc, SuperType, GenericParams, {}, {}, Frame, isBuiltin) {}

TypeBinding::TypeBinding(
    StringRef BindingName, mlir::Location Loc, const TypeBinding &SuperType,
    ParamsMap GenericParams, Flags BindingFlags, Frame Frame
)
    : TypeBinding(BindingName, Loc, SuperType, GenericParams, {}, {}, BindingFlags, Frame) {}

TypeBinding::TypeBinding(
    StringRef BindingName, mlir::Location Loc, const TypeBinding &SuperType,
    ParamsMap GenericParams, ParamsMap ConstructorParams, MembersMap Members, Frame Frame,
    bool IsBuiltin
)
    : TypeBinding(
          BindingName, Loc, SuperType, GenericParams, ConstructorParams, Members,
          Flags::MkBuiltin(IsBuiltin), Frame
      ) {}

TypeBinding::TypeBinding(
    StringRef BindingName, mlir::Location Loc, const TypeBinding &SuperType,
    ParamsMap GenericParams, ParamsMap ConstructorParams, MembersMap Members, Flags BindingFlags,
    Frame Frame
)
    : flags(BindingFlags), name(BindingName), loc(Loc), superType(&SuperType), members(Members),
      genericParams(GenericParams), constructorParams(ConstructorParams), frame(Frame) {}

TypeBinding::TypeBinding(
    uint64_t Value, mlir::Location Loc, const TypeBindings &Bindings, bool IsBuiltin
)
    : TypeBinding(Value, Loc, Bindings, Flags::MkBuiltin(IsBuiltin)) {}

TypeBinding::TypeBinding(
    uint64_t Value, mlir::Location Loc, const TypeBindings &Bindings, Flags BindingFlags
)
    : flags(BindingFlags), name(CONST), loc(Loc), constExpr(expr::ConstExpr::Val(Value)),
      superType(&Bindings.Get("Val")) {}

void TypeBinding::print(llvm::raw_ostream &os, bool fullPrintout) const {
  auto printType = [&]() {
    os << name.ref();
    if (flags.isSpecialized()) {
      getGenericParamsMapping().printParams(os, {.fullPrintout = false});
    } else {
      getGenericParamsMapping().printNames(os);
    }
    if (fullPrintout) {
      getConstructorParams().printParams(
          os, {.fullPrintout = false, .printIfEmpty = true, .header = '(', .footer = ')'}
      );
    }
    if (flags.isVariadic()) {
      os << "...";
    }
  };
  if (isConst()) {
    if (hasConstValue(constExpr)) {
      os << getConstValue(constExpr);
    } else {
      os << "?";
    }
    if (fullPrintout) {
      os << " : ";
      printType();
    }
  } else if (hasConstExpr()) {
    getConstExpr()->print(os);
    if (fullPrintout) {
      os << " : ";
      printType();
    }
  } else if (isGenericParam()) {
    os << name;
    if (fullPrintout) {
      os << " : ";
      printType();
    }
  } else {
    printType();
  }
  if (fullPrintout) {
    os << " { ";
    if (flags.isVariadic()) {
      os << "variadic ";
    }
    if (flags.isSpecialized()) {
      os << "specialized ";
    }
    if (flags.isSelfConstructor()) {
      os << "selfConstructor ";
    }
    if (flags.isBuiltin()) {
      os << "builtin ";
    }
    if (flags.isClosure()) {
      os << "closure ";
    }
    if (flags.isExtern()) {
      os << "extern ";
    }
    if (hasConstValue(constExpr)) {
      os << "const(" << getConstValue(constExpr) << ") ";
    }
    if (flags.isGenericParamName()) {
      os << "genericParam(" << name << ") ";
    }
    os << "constExpr(";
    if (constExpr) {
      constExpr->print(os);
    } else {
      os << "<<NULL>>";
    }
    os << ") ";
    os << "}";
    if (!getMembers().empty()) {
      os << " members { ";
      llvm::interleaveComma(getMembers(), os, [&](auto &member) {
        os << member.getKey() << ": ";
        auto &type = member.getValue();
        if (type.has_value()) {
          type->print(os, false);
        } else {
          os << "⊥";
        }
      });
      os << " }";
    }
    if (hasSuperType()) {
      os << " :> ";
      getSuperType().print(os, fullPrintout);
    }
  }
}

void TypeBinding::selfConstructs() {
  if (flags.isSelfConstructor()) {
    return;
  }
  flags.setSelfConstructor(true);
  constructorParams = ParamsMap().declare("x", *this);
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

  return superTypeIsEqual && flags == other.flags && name == other.name &&
         constExpr == other.constExpr && getMembers() == other.getMembers() &&
         getGenericParamsMapping() == other.getGenericParamsMapping() &&
         getConstructorParams() == other.getConstructorParams();
}

FailureOr<std::optional<TypeBinding>> TypeBinding::locateMember(StringRef memberName) const {
  const auto &membersMap = getMembers();
  auto it = membersMap.find(memberName);
  if (it != membersMap.end()) {
    return it->second;
  }
  if (superType != nullptr) {
    return superType->locateMember(memberName);
  }
  return failure();
}

//==-----------------------------------------------------------------------==//
// TypeBinding::ParamsStoragePtr
//==-----------------------------------------------------------------------==//

std::shared_ptr<ParamsStorage> TypeBinding::ParamsStorageFactory::init() {
  return ParamsStorage::Shared();
}

TypeBinding::ParamsStoragePtr &TypeBinding::ParamsStoragePtr::operator=(const ParamsMap &map) {
  set(ParamsStorage(map));
  return *this;
}
TypeBinding::ParamsStoragePtr &TypeBinding::ParamsStoragePtr::operator=(ParamsMap &&map) {
  set(ParamsStorage(map));
  return *this;
}

//==-----------------------------------------------------------------------==//
// TypeBinding::MembersMapFactory
//==-----------------------------------------------------------------------==//

std::shared_ptr<MembersMap> TypeBinding::MembersMapFactory::init() {
  return std::make_shared<MembersMap>();
}

TypeBinding::MembersMapPtr &TypeBinding::MembersMapPtr::operator=(const MembersMap &map) {
  set(map);
  return *this;
}
TypeBinding::MembersMapPtr &TypeBinding::MembersMapPtr::operator=(MembersMap &&map) {
  set(map);
  return *this;
}

//==-----------------------------------------------------------------------==//
// operator<< overloads
//==-----------------------------------------------------------------------==//

llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &os, const TypeBinding &type) {
  type.print(os);
  return os;
}

llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &os, const TypeBinding::Name &name) {
  os << name.ref();
  return os;
}

mlir::Diagnostic &zhl::operator<<(mlir::Diagnostic &diag, const TypeBinding &type) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  type.print(ss);
  diag << s;
  return diag;
}

mlir::Diagnostic &zhl::operator<<(mlir::Diagnostic &diag, const TypeBinding::Name &name) {
  diag << Twine(name.ref());
  return diag;
}

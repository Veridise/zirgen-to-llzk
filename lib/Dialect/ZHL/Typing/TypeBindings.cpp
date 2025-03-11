#include <cassert>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

#define DEBUG_TYPE "zhl-type-bindings"

using namespace zhl;
using namespace mlir;

const TypeBinding &TypeBindings::Component() {
  if (!Exists("Component")) {
    bindings.insert({"Component", TypeBinding(unk)});
  }
  return bindings.at("Component");
}

const TypeBinding &TypeBindings::Component() const { return bindings.at("Component"); }

const TypeBinding &TypeBindings::Bottom() const { return bottom; }

TypeBinding TypeBindings::Const(uint64_t value) const { return Const(value, unk); }

TypeBinding TypeBindings::Const(uint64_t value, Location loc) const {
  return TypeBinding(value, loc, *this);
}

TypeBinding TypeBindings::UnkConst() const { return UnkConst(unk); }

TypeBinding TypeBindings::UnkConst(Location loc) const {
  return TypeBinding(CONST, loc, Get("Val"));
}

TypeBinding TypeBindings::Array(TypeBinding type, uint64_t size) const {
  return Array(type, size, unk);
}

TypeBinding TypeBindings::Array(TypeBinding type, TypeBinding size) const {
  return Array(type, size, unk);
}

TypeBinding TypeBindings::Array(TypeBinding type, uint64_t size, Location loc) const {
  return Array(type, Const(size), loc);
}

TypeBinding TypeBindings::Array(TypeBinding type, TypeBinding size, Location loc) const {
  ParamsMap arrayGenericParams;
  auto cleanedType = TypeBinding::StripConst(type);
  arrayGenericParams.declare("T", cleanedType);
  arrayGenericParams.declare("N", size);
  TypeBinding array("Array", loc, Component(), arrayGenericParams, Frame(), true);
  array.specialized = true;
  return array;
}

TypeBinding TypeBindings::UnkArray(TypeBinding type) const { return UnkArray(type, unk); }

TypeBinding TypeBindings::UnkArray(TypeBinding type, Location loc) const {
  return Array(type, UnkConst(), loc);
}

bool TypeBindings::Exists(StringRef name) const { return bindings.find(name) != bindings.end(); }

const TypeBinding &TypeBindings::Get(StringRef name) const { return bindings.at(name); }

FailureOr<TypeBinding> TypeBindings::MaybeGet(StringRef name) const {
  if (Exists(name)) {
    return Get(name);
  }
  return mlir::failure();
}

TypeBinding &TypeBindings::Manage(const TypeBinding &binding) const {
  managedBindings.push_back(binding);
  return managedBindings.back();
}

TypeBindings::TypeBindings(Location defaultLoc)
    : unk(defaultLoc), bottom(TypeBinding(BOTTOM, unk, Component())) {
  (void)Component();
}

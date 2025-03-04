#include <cassert>
#include <iterator>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

using namespace zhl;
using namespace mlir;

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

TypeBinding TypeBindings::Array(TypeBinding type, TypeBinding size) const {
  return Array(type, size, unk);
}

zhl::TypeBinding
zhl::TypeBindings::Array(TypeBinding type, uint64_t size, mlir::Location loc) const {
  return Array(type, Const(size), loc);
}

TypeBinding TypeBindings::Array(TypeBinding type, TypeBinding size, mlir::Location loc) const {
  ParamsMap arrayGenericParams;
  arrayGenericParams.insert({{"T", 0}, type});
  arrayGenericParams.insert({{"N", 1}, size});
  TypeBinding array("Array", loc, Component(), arrayGenericParams, Frame(), true);
  array.specialized = true;
  return array;
}

zhl::TypeBinding zhl::TypeBindings::UnkArray(TypeBinding type) const { return UnkArray(type, unk); }

zhl::TypeBinding zhl::TypeBindings::UnkArray(TypeBinding type, mlir::Location loc) const {
  return Array(type, UnkConst(), loc);
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

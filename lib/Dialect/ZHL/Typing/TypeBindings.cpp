//===- TypeBindings.cpp - Type bindings factory -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <llvm/ADT/SmallString.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/ComponentSlot.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>
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
  TypeBinding::Flags flags;
  flags.setSpecialized().setBuiltin();
  TypeBinding array("Array", loc, Component(), arrayGenericParams, flags, Frame());
  array.selfConstructs();
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

static TypeBinding &&selfConstructs(TypeBinding &&t) {
  t.selfConstructs();
  return std::move(t);
}

TypeBindings::TypeBindings(Location defaultLoc)
    : unk(defaultLoc), bottom(TypeBinding(BOTTOM, unk, Component())) {
  auto trivial = [this](StringRef name) -> const TypeBinding & {
    return insert(name, selfConstructs(makeBuiltin(name, Component())));
  };

  auto &Val = trivial("Val");
  auto &Type = CreateBuiltin("Type", Component());

  trivial("ExtVal");
  trivial("String");

  auto T = zhl::TypeBinding::MakeGenericParam(Type, "T");
  auto N = zhl::TypeBinding::MakeGenericParam(Val, "N");

  insert(
      "Array", selfConstructs(makeBuiltin(
                   "Array", Component(), zhl::ParamsMap().declare("T", T).declare("N", N)
               ))
  );
}

const TypeBinding &TypeBindings::insert(StringRef name, TypeBinding &&binding) {
  assert(bindings.find(name) == bindings.end() && "double binding write");
  bindings.try_emplace(name, binding);
  return bindings.at(name);
}

std::string TypeBindings::NameUniquer::getName(llvm::StringRef originalName) {
  // If the name is new add it to the set and return it
  if (!seen.contains(originalName)) {
    seen.insert(originalName);
    return originalName.str();
  }

  // If it isn't new then try names until a new one is found.
  using S = llvm::SmallString<20>;
  S newName(originalName);
  // Format the new name by simply overwriting the addition made to the original name.
  // Since we are using a small string, for the majority of the cases these overwrites
  // should happen all the time in the stack.
  // If that's the case then only two copies should happen, when we add to the seen set
  // and when we allocate the returned string.
  auto format = [originalNameSize = originalName.size(), &newName](auto n) {
    auto name = "$_" + llvm::Twine(n);
    // Remove the last number (if any)
    if (newName.size() > originalNameSize) {
      newName.truncate(originalNameSize);
    }
    // Add the new number
    name.toVector(newName);
  };
  unsigned int n = 1;
  format(n);
  for (; seen.contains(newName); n++, format(n)) {
  }
  seen.insert(newName);
  return newName.str().str();
}

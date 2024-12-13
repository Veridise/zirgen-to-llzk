#pragma once

#include "ZirToZkir/Dialect/ZHL/Typing/TypeBindings.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LogicalResult.h>
#include <string_view>
#include <unordered_map>

namespace zhl {

class TypeBindings;

/// Binding to a ZIR type
class TypeBinding {
public:
  /// Returns the name of the type.
  std::string_view getName() const { return name; }

  void print(llvm::raw_ostream &os) const { os << name; }

  /// Returns true if the instance is a subtype of the argument
  mlir::LogicalResult subtypeOf(const TypeBinding &other) const {
    // TODO: Proper equality function
    if (getName() == other.getName()) {
      return mlir::success();
    }

    if (superType != nullptr) {
      return superType->subtypeOf(other);
    }
    return mlir::failure();
  }

  /// Returns the closest common supertype between the instance and the argument
  TypeBinding commonSupertypeWith(const TypeBinding &other) const {
    if (mlir::succeeded(subtypeOf(other))) {
      return other;
    }
    if (mlir::succeeded(other.subtypeOf(*this))) {
      return *this;
    }

    // TODO: Proper algorithm
    return TypeBinding();
  }

  // Creates an MLIR Type from the binding
  // mlir::Type materialize();
  // Attempts to create an specialized version of the type using the provided parameters.
  // mlir::FailureOr<TypeBinding> specialize(/*std::map<string, GenericParam>*/);

  TypeBinding() : name("Component"), superType(nullptr) {}
  TypeBinding(llvm::StringRef name, const TypeBinding &superType)
      : name(name), superType(&superType) {}

private:
  llvm::StringRef name;
  const TypeBinding *superType;
};

class TypeBindings {
public:
  const TypeBinding &Component() { return bindings["Component"]; }
  [[nodiscard]] bool Exists(std::string_view name) const {
    return bindings.find(name) != bindings.end();
  }

  const TypeBinding &Create(std::string_view name, const TypeBinding &superType) {
    assert(bindings.find(name) == bindings.end() && "double binding write");
    bindings[name] = TypeBinding(name, superType);
    return bindings[name];
  }

  [[nodiscard]] const TypeBinding &Get(std::string_view name) const { return bindings.at(name); }
  [[nodiscard]] mlir::FailureOr<TypeBinding> MaybeGet(std::string_view name) const {
    if (Exists(name)) {
      return Get(name);
    }
    return mlir::failure();
  }

private:
  inline const TypeBinding &
  GetOrCreate(llvm::StringRef name, std::function<TypeBinding()> factory) {
    if (bindings.find(name) == bindings.end()) {
      bindings[name] = factory();
    }
    return bindings[name];
  }
  std::unordered_map<std::string_view, TypeBinding> bindings;
};

} // namespace zhl

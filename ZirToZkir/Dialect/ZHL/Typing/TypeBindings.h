#pragma once

#include "llvm/ADT/StringRef.h"
#include <functional>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LogicalResult.h>
#include <string_view>
#include <unordered_map>

namespace zhl {

class TypeBindings;

#if 0
class TypeBindingW;

class TypeBindingBase {
public:
  virtual std::string_view getName() const = 0;
  void print(llvm::raw_ostream &os) const {
    os << getName();
    if (variadic) {
      os << "...";
    }
  }

  /// Returns true if the instance is a subtype of the argument
  virtual mlir::LogicalResult subtypeOf(const TypeBindingBase &other) const = 0;
  /// Returns the closest common supertype between the instance and the argument
  virtual const TypeBindingBase *commonSupertypeWith(const TypeBindingBase &other) const = 0;

private:
  bool variadic;
};

class TypeBinding {
private:
  class Concept {
public:
    virtual std::string_view getName() const = 0;
    virtual void print(llvm::raw_ostream &os) const = 0;
    /// Returns true if the instance is a subtype of the argument
    virtual mlir::LogicalResult subtypeOf(const Concept &other) const = 0;
    /// Returns the closest common supertype between the instance and the argument
    virtual TypeBinding commonSupertypeWith(const Concept &other) const = 0;
  };

  template<typename T>
  class Model : public Concept {
public:
      std::string_view getName() const final {
return inner.getName();
    }
     void print(llvm::raw_ostream &os) const final { inner->print(os); }
    /// Returns true if the instance is a subtype of the argument
     mlir::LogicalResult subtypeOf(const Concept &other) const final {
     return inner->subtypeOf(other.inner)
    }
    /// Returns the closest common supertype between the instance and the argument
     TypeBinding commonSupertypeWith(const TypeBinding &other) const = 0;

    private:
      const T *inner;
  };
};


class Component : public TypeBindingBase {
public:
  std::string_view getName() const final { return "Component"; }

  /// The root component is not a subtype of anything except itself.
  mlir::LogicalResult subtypeOf(const TypeBindingBase &other) const final {
    if (getName() == other.getName()) {
      return mlir::success();
    }
    return mlir::failure();
  }

  const TypeBindingBase *commonSupertypeWith(const TypeBindingBase &other) const final {
    return this;
  }
};
#endif

const std::string BOTTOM = "!";

/// Binding to a ZIR type
class TypeBinding {
public:
  /// Returns the name of the type.
  std::string_view getName() const { return name; }

  void print(llvm::raw_ostream &os) const {
    os << name;
    if (variadic) {
      os << "...";
    }
  }

  /// Returns true if the instance is a subtype of the argument
  mlir::LogicalResult subtypeOf(const TypeBinding &other) const {
    if (name == BOTTOM) {
      return mlir::success();
    }
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

  bool isBottom() const { return name == BOTTOM; }

  // Creates an MLIR Type from the binding
  // mlir::Type materialize();
  // Attempts to create an specialized version of the type using the provided parameters.
  // mlir::FailureOr<TypeBinding> specialize(/*std::map<string, GenericParam>*/);

  TypeBinding() : name("Component"), superType(nullptr) {}
  TypeBinding(llvm::StringRef name, const TypeBinding &superType)
      : name(name), superType(&superType) {}

  static TypeBinding WrapVariadic(const TypeBinding &t) {
    TypeBinding w = t;
    w.variadic = true;
    return w;
  }

private:
  bool variadic = false;
  llvm::StringRef name;
  const TypeBinding *superType;
};

class TypeBindings {
public:
  const TypeBinding &Component() { return bindings["Component"]; }
  const TypeBinding &Component() const { return bindings.at("Component"); }
  const TypeBinding &Bottom() const { return bottom; }
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
  TypeBinding bottom = TypeBinding(BOTTOM, Component());
};

} // namespace zhl

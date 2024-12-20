#pragma once

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringRef.h"
#include <deque>
#include <functional>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <map>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>
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
const std::string CONST = "$";

class TypeBinding;

using ParamsMap = std::map<std::pair<std::string_view, uint64_t>, TypeBinding>;
using MembersMap = std::map<std::string_view, std::optional<TypeBinding>>;

class Params {
public:
  using ParamsList = std::vector<TypeBinding>;
  using ParamNames = std::vector<std::string>;

  Params();
  Params(ParamsMap map);
  size_t size() const;

  operator ParamsMap() const;

  std::string_view getName(size_t i) const;

  TypeBinding getParam(size_t i) const;

  mlir::ArrayRef<std::string> getNames() const;

  void printNames(llvm::raw_ostream &os, char header = '<', char footer = '>') const;

  void printParams(llvm::raw_ostream &os, char header = '<', char footer = '>') const;

  ParamsList::iterator begin();
  ParamsList::const_iterator begin() const;
  ParamsList::iterator end();
  ParamsList::const_iterator end() const;

private:
  template <typename Elt>
  void print(
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

  ParamsList params;
  ParamNames names;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TypeBinding &type);

/// Binding to a ZIR type
class TypeBinding {
public:
  using ParamsList = std::vector<TypeBinding>;
  using ParamNames = std::vector<std::string>;
  /// Returns the name of the type.
  std::string_view getName() const;

  void print(llvm::raw_ostream &os) const;

  /// Returns true if the instance is a subtype of the argument
  mlir::LogicalResult subtypeOf(const TypeBinding &other) const;

  /// Returns the closest common supertype between the instance and the argument
  TypeBinding commonSupertypeWith(const TypeBinding &other) const;

  bool isBottom() const;
  bool isTypeMarker() const;
  bool isVal() const;
  bool isArray() const;
  bool isConst() const;
  bool isGeneric() const;
  /// Returns true if the type is not generic or has an specialization of its generic parameters
  bool isSpecialized() const;

  mlir::ArrayRef<std::string> getGenericParamNames() const;
  std::vector<mlir::Location> getConstructorParamLocations() const;
  const Params &getConstructorParams() const;

  mlir::FailureOr<TypeBinding> getArrayElement(std::function<mlir::InFlightDiagnostic()> emitError
  ) const;

  // Creates an MLIR Type from the binding
  // mlir::Type materialize();

  /// Attempts to create an specialized version of the type using the provided parameters.
  mlir::FailureOr<TypeBinding> specialize(
      std::function<mlir::InFlightDiagnostic()> emitError, mlir::ArrayRef<TypeBinding> params
  ) const;

  mlir::FailureOr<TypeBinding>
      getMember(mlir::StringRef, std::function<mlir::InFlightDiagnostic()>) const;

  TypeBinding(const TypeBinding &) = default;
  TypeBinding(TypeBinding &&) = default;
  TypeBinding &operator=(const TypeBinding &) = default;
  TypeBinding &operator=(TypeBinding &&) = default;
  TypeBinding(mlir::Location);
  TypeBinding(llvm::StringRef name, mlir::Location loc, const TypeBinding &superType);
  TypeBinding(
      llvm::StringRef name, mlir::Location loc, const TypeBinding &superType,
      ParamsMap t_genericParams
  );
  TypeBinding(
      llvm::StringRef name, mlir::Location loc, const TypeBinding &superType,
      ParamsMap t_genericParams, ParamsMap t_constructorParams, MembersMap members
  );
  TypeBinding(uint64_t value, mlir::Location loc, const TypeBindings &bindings);
  TypeBinding WithUpdatedLocation(mlir::Location loc) const;

  static TypeBinding WrapVariadic(const TypeBinding &t);

  friend TypeBindings;

private:
  bool variadic = false;
  bool specialized = false;
  llvm::StringRef name;
  mlir::Location loc;
  std::optional<uint64_t> constVal;
  const TypeBinding *superType;
  MembersMap members;
  Params genericParams;
  Params constructorParams;
};

class TypeBindings {
public:
  explicit TypeBindings(mlir::OpBuilder &);

  const TypeBinding &Component();
  const TypeBinding &Component() const;
  const TypeBinding &Bottom() const;
  TypeBinding Const(uint64_t value, mlir::Location loc) const;
  TypeBinding UnkConst(mlir::Location loc) const;
  TypeBinding Array(TypeBinding type, uint64_t size, mlir::Location loc) const;
  TypeBinding UnkArray(TypeBinding type, mlir::Location loc) const;

  TypeBinding Const(uint64_t value) const;
  TypeBinding UnkConst() const;
  TypeBinding Array(TypeBinding type, uint64_t size) const;
  TypeBinding UnkArray(TypeBinding type) const;

  [[nodiscard]] bool Exists(std::string_view name) const;

  template <typename... Args>
  const TypeBinding &Create(std::string_view name, mlir::Location loc, Args &&...args) {
    assert(bindings.find(name) == bindings.end() && "double binding write");
    bindings.emplace(name, TypeBinding(name, loc, std::forward<Args>(args)...));
    return bindings.at(name);
  }

  template <typename... Args> const TypeBinding &Create(std::string_view name, Args &&...args) {
    return Create(name, unk, std::forward<Args>(args)...);
  }

  [[nodiscard]] const TypeBinding &Get(std::string_view name) const;
  [[nodiscard]] mlir::FailureOr<TypeBinding> MaybeGet(std::string_view name) const;
  [[nodiscard]] const TypeBinding &Manage(const TypeBinding &);

private:
  mlir::Location unk;
  std::unordered_map<std::string_view, TypeBinding> bindings;
  std::deque<TypeBinding> managedBindings;
  TypeBinding bottom;
};

} // namespace zhl

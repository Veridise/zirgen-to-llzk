#pragma once

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <deque>
#include <functional>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <map>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string_view>
#include <unordered_map>

namespace zhl {

class TypeBindings;

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
  mlir::ArrayRef<TypeBinding> getParams() const;

  const TypeBinding *operator[](std::string_view name) const;
  TypeBinding *operator[](std::string_view name);

  void printMapping(llvm::raw_ostream &os) const;
  void printNames(llvm::raw_ostream &os, char header = '<', char footer = '>') const;
  void printParams(llvm::raw_ostream &os, char header = '<', char footer = '>') const;

  ParamsList::iterator begin();
  ParamsList::const_iterator begin() const;
  ParamsList::iterator end();
  ParamsList::const_iterator end() const;

  bool empty() const;

  void replaceParam(std::string_view name, const TypeBinding &binding);

  bool operator==(const Params &) const;

private:
  template <typename Elt>
  void print(
      const std::vector<Elt> &lst, llvm::raw_ostream &os, std::function<void(const Elt &)> handler,
      char header, char footer
  ) const;

  ParamsList params;
  ParamNames names;
};

/// Interface for the different types of bindings
class TypeBindingImpl {
public:
  virtual ~TypeBindingImpl() = default;
  virtual std::string_view getName() const = 0;
};

/// The root Component
class ComponentRoot : public TypeBindingImpl {
public:
  virtual std::string_view getName() const override { return "Component"; }
};

enum class TrivialTypes { Val, String, Array };

/// Represents the trivial types Val, String, Array, etc.
class TrivialType : public TypeBindingImpl {
public:
  virtual std::string_view getName() const override {
    switch (type) {
    case zhl::TrivialTypes::Val:
      return "Val";
    case zhl::TrivialTypes::String:
      return "String";
    case zhl::TrivialTypes::Array:
      return "Array";
    }
  }

private:
  TrivialTypes type;
};

/// Represents the bottom type that is a subtype of any other.
class BottomType : public TypeBindingImpl {
public:
  virtual std::string_view getName() const override { return BOTTOM; }
};

/// Represents the type of a component.
class NamedBinding : public TypeBindingImpl {
public:
  virtual std::string_view getName() const override { return name; }

private:
  std::string name;
  MembersMap members;
  Params genericParams;
  Params constructorParams;
  const TypeBinding *superType;
};

/// Represents a type variable.
class TypeVariable : public TypeBindingImpl {
public:
  virtual std::string_view getName() const override { return varName; }

private:
  std::string varName;
};

/// Represents a constant Val.
class ConstVal : public TypeBindingImpl {
public:
  virtual std::string_view getName() const override { return valueStrRepr; }

private:
  uint64_t value;
  std::string valueStrRepr;
};

/// Represents a constant Val whose actual value is unknown.
class UnkConstVal : public TypeBindingImpl {
public:
  virtual std::string_view getName() const override {
    if (varName.has_value()) {
      return *varName;
    } else {
      return "?";
    }
  }

private:
  std::optional<std::string> varName;
};

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
  bool isKnownConst() const;
  bool isGeneric() const;
  bool isGenericParam() const;
  bool isBuiltin() const;
  /// Returns true if the type is not generic or has an specialization of its generic parameters
  bool isSpecialized() const;
  bool isVariadic() const;
  bool hasSuperType() const;

  mlir::ArrayRef<std::string> getGenericParamNames() const;
  mlir::ArrayRef<TypeBinding> getGenericParams() const;
  std::vector<mlir::Location> getConstructorParamLocations() const;
  const Params &getConstructorParams() const;
  const Params &getGenericParamsMapping() const;
  const MembersMap &getMembers() const;
  mlir::Location getLocation() const;
  const TypeBinding &getSuperType() const;
  TypeBinding &getSuperType();
  uint64_t getConst() const;
  llvm::StringRef getGenericParamName() const;

  mlir::FailureOr<TypeBinding> getArrayElement(std::function<mlir::InFlightDiagnostic()> emitError
  ) const;

  void replaceGenericParamByName(std::string_view name, const TypeBinding &binding);

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
  TypeBinding(
      llvm::StringRef name, mlir::Location loc, const TypeBinding &superType, bool isBuiltin = false
  );
  TypeBinding(
      llvm::StringRef name, mlir::Location loc, const TypeBinding &superType,
      ParamsMap t_genericParams, bool isBuiltin = false
  );
  TypeBinding(
      llvm::StringRef name, mlir::Location loc, const TypeBinding &superType,
      ParamsMap t_genericParams, ParamsMap t_constructorParams, MembersMap members,
      bool isBuiltin = false
  );
  TypeBinding(
      uint64_t value, mlir::Location loc, const TypeBindings &bindings, bool isBuiltin = false
  );
  TypeBinding WithUpdatedLocation(mlir::Location loc) const;

  static TypeBinding WrapVariadic(const TypeBinding &t);
  static TypeBinding MakeGenericParam(const TypeBinding &t, llvm::StringRef name);

  friend TypeBindings;

  void selfConstructs();
  void markAsSpecialized();

  bool operator==(const TypeBinding &) const;

private:
  mlir::FailureOr<std::optional<TypeBinding>> locateMember(mlir::StringRef) const;

  bool variadic = false;
  bool specialized = false;
  bool selfConstructor = false;
  bool builtin = false;
  llvm::StringRef name;
  mlir::Location loc;
  std::optional<uint64_t> constVal;
  std::optional<llvm::StringRef> genericParamName;
  TypeBinding *superType;
  MembersMap members;
  Params genericParams;
  Params constructorParams;

  std::shared_ptr<TypeBindingImpl> impl;
};

class TypeBindings {
public:
  explicit TypeBindings(mlir::Location);

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
    llvm::dbgs() << "Creating binding with name " << name << "\n";
    assert(bindings.find(name) == bindings.end() && "double binding write");
    bindings.emplace(name, TypeBinding(name, loc, std::forward<Args>(args)...));
    return bindings.at(name);
  }

  template <typename... Args> const TypeBinding &Create(std::string_view name, Args &&...args) {
    return Create(name, unk, std::forward<Args>(args)...);
  }

  template <typename... Args>
  const TypeBinding &CreateBuiltin(std::string_view name, mlir::Location loc, Args &&...args) {
    llvm::dbgs() << "Creating binding with name " << name << "\n";
    assert(bindings.find(name) == bindings.end() && "double binding write");
    bindings.emplace(name, TypeBinding(name, loc, std::forward<Args>(args)..., true));
    return bindings.at(name);
  }

  template <typename... Args>
  const TypeBinding &CreateBuiltin(std::string_view name, Args &&...args) {
    return CreateBuiltin(name, unk, std::forward<Args>(args)...);
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

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const zhl::TypeBinding &b);

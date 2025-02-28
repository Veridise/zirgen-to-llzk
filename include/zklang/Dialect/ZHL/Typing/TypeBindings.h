#pragma once

#include <cassert>
#include <deque>
#include <functional>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <map>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string_view>
#include <unordered_map>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>

namespace zhl {

class TypeBindings;

const std::string BOTTOM = "!";
const std::string CONST = "$";

class TypeBinding;

using ParamsMap = std::map<std::pair<std::string_view, uint64_t>, TypeBinding>;
using MembersMap = std::map<std::string_view, std::optional<TypeBinding>>;
using EmitErrorFn = llvm::function_ref<mlir::InFlightDiagnostic()>;

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

/// Binding to a ZIR type
class TypeBinding {
public:
  using ParamsList = std::vector<TypeBinding>;
  using ParamNames = std::vector<std::string>;
  /// Returns the name of the type.
  std::string_view getName() const;
  void setName(mlir::StringRef);

  void print(llvm::raw_ostream &os, bool fullPrintout = false) const;

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
  Params &getConstructorParams();
  const Params &getGenericParamsMapping() const;
  Params &getGenericParamsMapping();
  const MembersMap &getMembers() const;
  MembersMap &getMembers();
  mlir::Location getLocation() const;
  const TypeBinding &getSuperType() const;
  TypeBinding &getSuperType();
  uint64_t getConst() const;
  llvm::StringRef getGenericParamName() const;

  mlir::FailureOr<TypeBinding> getArrayElement(EmitErrorFn emitError) const;
  mlir::FailureOr<TypeBinding> getArraySize(EmitErrorFn emitError) const;

  /// Returns the type of the concrete array type this binding supports. Either because the binding
  /// itself is an array type or because one of the types in the super chain is an Array type.
  mlir::FailureOr<TypeBinding> getConcreteArrayType() const;

  void replaceGenericParamByName(std::string_view name, const TypeBinding &binding);

  /// Attempts to create an specialized version of the type using the provided parameters.
  mlir::FailureOr<TypeBinding> specialize(
      std::function<mlir::InFlightDiagnostic()> emitError, mlir::ArrayRef<TypeBinding> params
  ) const;

  mlir::FailureOr<TypeBinding>
      getMember(mlir::StringRef, std::function<mlir::InFlightDiagnostic()>) const;

  TypeBinding(const TypeBinding &);
  TypeBinding(TypeBinding &&);
  TypeBinding &operator=(const TypeBinding &);
  TypeBinding &operator=(TypeBinding &&);
  TypeBinding(mlir::Location);
  TypeBinding(
      llvm::StringRef name, mlir::Location loc, const TypeBinding &superType, Frame frame = Frame(),
      bool isBuiltin = false
  );
  TypeBinding(
      llvm::StringRef name, mlir::Location loc, const TypeBinding &superType,
      ParamsMap t_genericParams, Frame frame = Frame(), bool isBuiltin = false
  );
  TypeBinding(
      llvm::StringRef name, mlir::Location loc, const TypeBinding &superType,
      ParamsMap t_genericParams, ParamsMap t_constructorParams, MembersMap members,
      Frame frame = Frame(), bool isBuiltin = false
  );
  TypeBinding(
      uint64_t value, mlir::Location loc, const TypeBindings &bindings, bool isBuiltin = false
  );
  TypeBinding WithUpdatedLocation(mlir::Location loc) const;
  TypeBinding ReplaceFrame(Frame) const;

  static TypeBinding WrapVariadic(const TypeBinding &t);
  static TypeBinding MakeGenericParam(const TypeBinding &t, llvm::StringRef name);
  static const TypeBinding &StripConst(const TypeBinding &);
  static TypeBinding WithClosure(const TypeBinding &);
  static TypeBinding WithoutClosure(const TypeBinding &);

  friend TypeBindings;
  friend mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, const zhl::TypeBinding &b);

  void selfConstructs();
  void markAsSpecialized();

  bool operator==(const TypeBinding &) const;

  void markSlot(FrameSlot *);
  FrameSlot *getSlot() const;

  Frame getFrame() const;

  bool hasClosure() const;

private:
  mlir::FailureOr<std::optional<TypeBinding>> locateMember(mlir::StringRef) const;

  bool variadic = false;
  bool specialized = false;
  bool selfConstructor = false;
  bool builtin = false;
  bool closure = false;
  llvm::StringRef name;
  mlir::Location loc;
  std::optional<uint64_t> constVal;
  std::optional<llvm::StringRef> genericParamName;
  TypeBinding *superType;
  MembersMap members;
  Params genericParams;
  Params constructorParams;
  Frame frame;
  FrameSlot *slot = nullptr;
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
  TypeBinding Array(TypeBinding type, TypeBinding size, mlir::Location loc) const;
  TypeBinding UnkArray(TypeBinding type, mlir::Location loc) const;

  TypeBinding Const(uint64_t value) const;
  TypeBinding UnkConst() const;
  TypeBinding Array(TypeBinding type, uint64_t size) const;
  TypeBinding Array(TypeBinding type, TypeBinding size) const;
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

  template <typename... Args>
  const TypeBinding &CreateBuiltin(std::string_view name, mlir::Location loc, Args &&...args) {
    assert(bindings.find(name) == bindings.end() && "double binding write");
    bindings.emplace(name, TypeBinding(name, loc, std::forward<Args>(args)..., Frame(), true));
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

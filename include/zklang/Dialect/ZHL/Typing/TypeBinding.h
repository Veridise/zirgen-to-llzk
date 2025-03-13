#pragma once

#include <cassert>
#include <cstdint>
#include <deque>
#include <functional>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Debug.h>
#include <map>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string_view>
#include <unordered_map>
#include <zklang/Dialect/ZHL/Typing/Expr.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Support/CopyablePointer.h>

namespace llvm {
class raw_ostream;
}

namespace zhl {

const mlir::StringRef BOTTOM = "!";
const mlir::StringRef CONST = "$";

class TypeBindings;
class TypeBinding;
struct ParamData;

using MembersMap = llvm::StringMap<std::optional<TypeBinding>>;
using EmitErrorFn = llvm::function_ref<mlir::InFlightDiagnostic()>;
using ParamName = std::string;
using ParamsList = mlir::SmallVector<TypeBinding, 0>;
using ParamNames = mlir::SmallVector<ParamName>;

struct ParamsStorage;
struct ParamsMap;
class Params;
class MutableParams;

/// Binding to a ZIR type
class TypeBinding {
private:
  struct ParamsStorageFactory {
    /// Initializes empty storage
    static ParamsStorage *init();
  };

  struct ParamsStoragePtr : public zklang::CopyablePointer<ParamsStorage, ParamsStorageFactory> {
    using zklang::CopyablePointer<ParamsStorage, ParamsStorageFactory>::CopyablePointer;

    ParamsStoragePtr(ParamsMap &);
    operator MutableParams();
    operator Params() const;

    ParamsStoragePtr &operator=(ParamsMap &);
    ParamsStoragePtr &operator=(ParamsMap &&);
  };

public:
  class Name {
  public:
    Name(const Name &);
    Name(Name &&);
    Name &operator=(const Name &);
    Name &operator=(Name &&);

    Name(mlir::StringRef);
    ~Name();

    Name &operator=(mlir::StringRef);
    operator mlir::StringRef() const;

    bool operator==(const Name &) const;
    bool operator==(mlir::StringRef) const;

    mlir::StringRef ref() const;

    friend mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, const Name &);

  private:
    struct Impl;

    std::shared_ptr<Impl> impl;
  };

  /// Returns the name of the type.
  mlir::StringRef getName() const;
  void setName(mlir::StringRef);

  void print(llvm::raw_ostream &os, bool fullPrintout = false) const;

  /// Returns true if the instance is a subtype of the argument
  mlir::LogicalResult subtypeOf(const TypeBinding &other) const;

  /// Returns the closest common supertype between the instance and the argument
  TypeBinding commonSupertypeWith(const TypeBinding &other) const;

  bool isBottom() const;
  bool isTypeMarker() const;
  bool isVal() const;
  bool isTransitivelyVal() const;
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

  mlir::ArrayRef<ParamName> getGenericParamNames() const;
  mlir::MutableArrayRef<TypeBinding> getGenericParams();
  mlir::ArrayRef<TypeBinding> getGenericParams() const;
  mlir::SmallVector<TypeBinding, 0> getDeclaredGenericParams() const;
  mlir::SmallVector<mlir::Location> getConstructorParamLocations() const;
  Params getConstructorParams() const;
  MutableParams getConstructorParams();
  Params getGenericParamsMapping() const;
  MutableParams getGenericParamsMapping();
  const MembersMap &getMembers() const;
  MembersMap &getMembers();
  mlir::Location getLocation() const;
  const TypeBinding &getSuperType() const;
  void setSuperType(TypeBinding &);
  uint64_t getConst() const;
  llvm::StringRef getGenericParamName() const;

  mlir::FailureOr<TypeBinding> getArrayElement(EmitErrorFn emitError) const;
  mlir::FailureOr<TypeBinding> getArraySize(EmitErrorFn emitError) const;

  /// Returns the type of the concrete array type this binding supports. Either because the binding
  /// itself is an array type or because one of the types in the super chain is an Array type.
  mlir::FailureOr<TypeBinding> getConcreteArrayType() const;

  void replaceGenericParamByName(mlir::StringRef name, const TypeBinding &binding);

  /// Attempts to create an specialized version of the type using the provided parameters.
  mlir::FailureOr<TypeBinding>
  specialize(EmitErrorFn emitError, mlir::ArrayRef<TypeBinding> params, TypeBindings &) const;

  mlir::FailureOr<TypeBinding> getMember(mlir::StringRef, EmitErrorFn) const;

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
  ~TypeBinding();

  static TypeBinding WrapVariadic(const TypeBinding &t);
  static TypeBinding MakeGenericParam(const TypeBinding &t, llvm::StringRef name);
  static TypeBinding WithExpr(const TypeBinding &, expr::ConstExpr);
  static TypeBinding NoExpr(const TypeBinding &);
  static const TypeBinding &StripConst(const TypeBinding &);
  static TypeBinding WithClosure(const TypeBinding &);
  static TypeBinding WithoutClosure(const TypeBinding &);

  friend TypeBindings;
  friend mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, const TypeBinding &b);

  void selfConstructs();
  void markAsSpecialized();

  bool operator==(const TypeBinding &) const;

  void markSlot(FrameSlot *);
  FrameSlot *getSlot() const;

  Frame getFrame() const;

  bool hasConstExpr() const;
  const expr::ConstExpr &getConstExpr() const;
  void setConstExpr(expr::ConstExpr);
  bool hasClosure() const;

private:
  mlir::FailureOr<std::optional<TypeBinding>> locateMember(mlir::StringRef) const;

  bool variadic = false;
  bool specialized = false;
  bool selfConstructor = false;
  bool builtin = false;
  bool closure = false;
  Name name;
  mlir::Location loc;
  expr::ConstExpr constExpr;
  std::optional<std::string> genericParamName;
  const TypeBinding *superType;
  MembersMap members;
  ParamsStoragePtr genericParams;
  ParamsStoragePtr constructorParams;
  Frame frame;
  FrameSlot *slot = nullptr;
};

} // namespace zhl

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const zhl::TypeBinding::Name &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const zhl::TypeBinding &b);

//===- TypeBinding.h - Type information & metadata --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes the TypeBinding class that serves as the central
// datatype for the type analysis module.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <cstdint>
#include <llvm/ADT/Bitset.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Debug.h>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/Expr.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/Params.h>
#include <zklang/Support/COW.h>

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
    static std::shared_ptr<ParamsStorage> init();
  };

  struct ParamsStoragePtr : public zklang::COW<ParamsStorage, ParamsStorageFactory> {
    using zklang::COW<ParamsStorage, ParamsStorageFactory>::COW;

    operator Params() const { return Params(get()); }
    operator MutableParams() { return MutableParams(get()); }

    ParamsStoragePtr &operator=(const ParamsMap &);
    ParamsStoragePtr &operator=(ParamsMap &&);
  };

  struct MembersMapFactory {
    /// Initializes empty storage
    static std::shared_ptr<MembersMap> init();
  };

  struct MembersMapPtr : public zklang::COW<MembersMap, MembersMapFactory> {
    using zklang::COW<MembersMap, MembersMapFactory>::COW;

    MembersMapPtr &operator=(const MembersMap &);
    MembersMapPtr &operator=(MembersMap &&);
  };

public:
  //==---------------------------------------------------------------------==//
  // Type binding name
  //==---------------------------------------------------------------------==//

  /// A shared pointer to the name of the type binding. Any type binding that is copied will have a
  /// reference to the same name. This is done for propagating type renames to all bindings that
  /// reference the same type.
  class Name {
    using Impl = llvm::SmallString<10>;
    std::shared_ptr<Impl> impl;

  public:
    Name(const Name &) = default;
    Name(Name &&) = default;
    Name &operator=(const Name &) = default;
    Name &operator=(Name &&) = default;

    Name(mlir::StringRef name) : impl(std::make_shared<Impl>(name)) {}
    ~Name() = default;

    Name &operator=(mlir::StringRef newName) {
      *impl = newName;
      return *this;
    }

    operator mlir::StringRef() const { return *impl; }

    bool operator==(const Name &other) const { return ref() == other.ref(); }

    bool operator==(mlir::StringRef other) const { return ref() == other; }

    mlir::StringRef ref() const { return *impl; }
  };

  /// Returns the name of the type.
  mlir::StringRef getName() const { return name.ref(); }

  /// Sets the name of the type and of any type it shares its name with.
  void setName(mlir::StringRef);

  //==---------------------------------------------------------------------==//
  // Type binding flags
  //==---------------------------------------------------------------------==//

  class Flags {
    enum : unsigned {
      /// A variadic type binding represents a constructor argument that can accept any number of
      /// arguments.
      Variadic,
      /// A type binding is considered specialized if it has types or values assigned to its generic
      /// parameters. This flag marks the type binding as having done that transformation.
      Specialized,
      /// Marks the type binding as having been declared to self construct.
      SelfConstructor,
      /// Marks the type binding as a language builtin type.
      Builtin,
      /// Marks the type binding as a closure that needs to have a component created during
      /// lowering.
      Closure,
      /// Marks the type binding as an external component.
      Extern,
      /// Marks the type binding as needing back-variable support.
      NeedsBackVariables,
      /// Marks the type binding as a generic parameter name.
      GenericParamName,

      Flags_End
    };

    using bitset = llvm::Bitset<Flags_End>;

    bitset flags;

    constexpr void set(unsigned idx, bool value) {
      if (value) {
        flags.set(idx);
      } else {
        flags.reset(idx);
      }
    }

  public:
    constexpr Flags() {}
    constexpr Flags(std::initializer_list<unsigned> Init) : flags(Init) {}

    static constexpr Flags MkBuiltin(bool builtin) {
      Flags f;
      f.setBuiltin(builtin);
      return f;
    }

    constexpr bool operator==(const Flags &other) const { return flags == other.flags; }

    /// Returns true if the type binding has been marked as variadic.
    constexpr bool isVariadic() const { return flags[Variadic]; }
    /// Sets the Variadic flag.
    constexpr Flags &setVariadic(bool B = true) {
      set(Variadic, B);
      return *this;
    }

    /// Returns true if the type has an specialization of its generic parameters.
    constexpr bool isSpecialized() const { return flags[Specialized]; }
    /// Sets the Specialized flag.
    constexpr Flags &setSpecialized(bool B = true) {
      set(Specialized, B);
      return *this;
    }

    /// Returns true if the type binding has been marked as self constructing.
    constexpr bool isSelfConstructor() const { return flags[SelfConstructor]; }
    /// Sets the SelfConstructor flag.
    constexpr Flags &setSelfConstructor(bool B = true) {
      set(SelfConstructor, B);
      return *this;
    }

    /// Returns true if this type binding has a type that is a language builtin.
    constexpr bool isBuiltin() const { return flags[Builtin]; }
    /// Sets the Builtin flag.
    constexpr Flags &setBuiltin(bool B = true) {
      set(Builtin, B);
      return *this;
    }

    /// Returns true if the type binding has been marked as a closure that needs to have a component
    /// definition op generated.
    constexpr bool isClosure() const { return flags[Closure]; }
    /// Sets the Closure flag.
    constexpr Flags &setClosure(bool B = true) {
      set(Closure, B);
      return *this;
    }

    /// Return true if the type binding is of a external component.
    constexpr bool isExtern() const { return flags[Extern]; }
    /// Sets the Extern flag.
    constexpr Flags &setExtern(bool B = true) {
      set(Extern, B);
      return *this;
    }

    /// Return true if the type binding needs back-variable support.
    constexpr bool needsBackVariables() const { return flags[NeedsBackVariables]; }
    /// Sets the Extern flag.
    constexpr Flags &setBackVariablesNeed(bool B = true) {
      set(NeedsBackVariables, B);
      return *this;
    }

    /// Return true if the type binding is a generic parameter name.
    constexpr bool isGenericParamName() const { return flags[GenericParamName]; }
    /// Sets the GenericParamName flag.
    constexpr Flags &setGenericParamName(bool B = true) {
      set(GenericParamName, B);
      return *this;
    }
  };

  //==---------------------------------------------------------------------==//
  // Type binding properties
  //==---------------------------------------------------------------------==//

  /// Returns true if the instance is a subtype of the argument
  mlir::LogicalResult subtypeOf(const TypeBinding &other) const;

  /// Returns the closest common supertype between the instance and the argument
  TypeBinding commonSupertypeWith(const TypeBinding &other) const;

  /// Returns true if the type binding is the bottom type.
  bool isBottom() const { return name.ref() == BOTTOM; }

  /// Returns true if the type binding is of type 'Type', which represents that a generic parameter
  /// is expecting a type when specialized.
  bool isTypeMarker() const { return name.ref() == "Type"; }

  /// Returns true if the type binding is of type 'Val'.
  bool isVal() const { return name.ref() == "Val"; }

  /// Returns true if one of the type binding's super types is the 'Val' type.
  bool isTransitivelyVal() const;

  /// Returns true if this type binding is of Array type or if one of its super types is.
  bool isArray() const;

  /// Returns true if this type binding is a constant value of type 'Val'.
  bool isConst() const { return name.ref() == CONST; }

  /// Returns true if this type binding is a constant value of type 'Val' of which we know its
  /// integer value.
  bool isKnownConst() const;

  /// Returns true if the type binding has generic parameters.
  bool isGeneric() const { return getGenericParamsMapping().size() > 0; }

  /// Returns true if the type binding represents a generic parameter;
  bool isGenericParam() const;

  /// Returns true if this type binding has a type that is a language builtin.
  bool isBuiltin() const { return flags.isBuiltin(); }

  /// Returns true if the type is not generic or has an specialization of its generic parameters.
  bool isSpecialized() const { return !isGeneric() || flags.isSpecialized(); }

  /// Returns true if the type binding has been marked as variadic.
  bool isVariadic() const { return flags.isVariadic(); }

  /// Returns true if the type binding has a super type.
  bool hasSuperType() const { return superType != nullptr; }

  /// Return true if the type binding is of a external component.
  bool isExtern() const { return flags.isExtern(); }

  /// Return true if the type binding needs back-variable support.
  bool needsBackVariables() const { return flags.needsBackVariables(); }

  /// Returns a reference to the super type of this type binding. Aborts if the
  /// type binding does not have a super type.
  const TypeBinding &getSuperType() const {
    assert(superType != nullptr);
    return *superType;
  }

  /// Sets the super type of the type binding.
  void setSuperType(TypeBinding &newSuperType) { superType = &newSuperType; }

  /// Sets the slot of the type binding. If the current slot is equal to the new slot this member
  /// function is a no-op. If the current slot is not null it can only be overwriten with nullptr.
  void markSlot(FrameSlot *);

  /// Returns the slot this type binding will use if it needs to define memory in the component.
  /// Returns nullptr if the type binding does not have any associated slot.
  FrameSlot *getSlot() const { return slot; }

  /// Returns the frame this type binding defines as the memory in a component it will need.
  Frame getFrame() const { return frame; }

  /// Returns true if the type binding is has an associated constant expression.
  bool hasConstExpr() const { return bool(constExpr); }

  /// Returns a reference to the constant expression. If the type binding does not have one the
  /// value it returns is falsey.
  const expr::ConstExpr &getConstExpr() const { return constExpr; };

  /// Sets the constant expression of the type binding.
  void setConstExpr(expr::ConstExpr expr) { constExpr = expr; };

  /// Returns true if the type binding has been marked as a closure that needs to have a component
  /// definition op generated.
  bool hasClosure() const { return flags.isClosure(); }

  /// Returns the associated location of this type binding.
  mlir::Location getLocation() const { return loc; }

  /// If the type binding holds a constant expression that comprises a single literal number returns
  /// its integer representation.
  uint64_t getConst() const;

  //==---------------------------------------------------------------------==//
  // Generic perameters
  //==---------------------------------------------------------------------==//

  /// Returns a view of the generic parameters.
  Params getGenericParamsMapping() const { return genericParams; }

  /// Returns mutable a view of the generic parameters.
  MutableParams getGenericParamsMapping() { return genericParams; }

  /// Returns the names of the generic parameters of the type binding.
  mlir::ArrayRef<ParamName> getGenericParamNames() const {
    return getGenericParamsMapping().getNames();
  }

  /// Returns a mutable array of the types of the generic parameters of the type binding.
  mlir::MutableArrayRef<TypeBinding> getGenericParams() {
    return getGenericParamsMapping().getParams();
  }

  /// Returns an array of the types of the generic parameters of the type binding.
  mlir::ArrayRef<TypeBinding> getGenericParams() const {
    return getGenericParamsMapping().getParams();
  }

  /// Returns a vector of the types of the generic parameters that have been declared in the source
  /// code.
  mlir::SmallVector<TypeBinding, 0> getDeclaredGenericParams() const {
    return getGenericParamsMapping().getDeclaredParams();
  }

  /// If the type binding is a generic parameter returns the name of the parameter.
  /// If the type binding is not this method aborts.
  mlir::StringRef getGenericParamName() const {
    assert(isGenericParam());
    return name;
  }

  /// Locates a generic parameter by name and sets its type binding to a copy of the given one.
  void replaceGenericParamByName(mlir::StringRef paramName, const TypeBinding &binding) {
    getGenericParamsMapping().replaceParam(paramName, binding);
  }

  /// If this matches a parameter in the given set returns the associated binding. Otherwise returns
  /// a copy of itself. Returns failure() if the type binding is a const expression and its
  /// internal mapping failed.
  mlir::FailureOr<TypeBinding> map(Params, EmitErrorFn) const;

  //==---------------------------------------------------------------------==//
  // Constructor parameters
  //==---------------------------------------------------------------------==//

  /// Returns a view of the constructor parameters.
  Params getConstructorParams() const { return constructorParams; }

  /// Returns mutable a view of the constructor parameters.
  MutableParams getConstructorParams() { return constructorParams; }

  /// Returns a vector with the source locations associated to the types of the constructor
  /// parameters of the type binding.
  mlir::SmallVector<mlir::Location> getConstructorParamLocations() const;

  //==---------------------------------------------------------------------==//
  // Members
  //==---------------------------------------------------------------------==//

  /// Returns a constant reference to the members defined in the type binding.
  const MembersMap &getMembers() const { return *members; }

  /// Returns a reference to the members defined in the type binding.
  MembersMap &getMembersMut() { return *members; }

  /// Attempts to find an accesible member of the type binding by name. Returns failure if the
  /// member is not public or it doesn't exist.
  mlir::FailureOr<TypeBinding> getMember(mlir::StringRef, EmitErrorFn) const;

  //==---------------------------------------------------------------------==//
  // Array helper methods
  //==---------------------------------------------------------------------==//

  /// Returns the type binding that corresponds to the inner type of the Array builtin. Returns
  /// failure if the type binding cannot be used as an Array.
  mlir::FailureOr<TypeBinding> getArrayElement(EmitErrorFn emitError) const;

  /// Returns the type binding that corresponds to the size of the Array builtin. Returns failure if
  /// the type binding cannot be used as an Array.
  mlir::FailureOr<TypeBinding> getArraySize(EmitErrorFn emitError) const;

  /// Returns the type of the concrete array type this binding supports. Either because the binding
  /// itself is an array type or because one of the types in the super chain is an Array type.
  mlir::FailureOr<TypeBinding> getConcreteArrayType() const;

  //==---------------------------------------------------------------------==//
  // Constructors & destructors
  //==---------------------------------------------------------------------==//

  TypeBinding(const TypeBinding &) = default;
  TypeBinding(TypeBinding &&) = default;
  TypeBinding &operator=(const TypeBinding &) = default;
  TypeBinding &operator=(TypeBinding &&) = default;

  /// Constructs a type binding of type Component.
  TypeBinding(mlir::Location);

  TypeBinding(
      mlir::StringRef Name, mlir::Location Loc, const TypeBinding &SuperType, Frame = Frame(),
      bool IsBuiltin = false
  );
  TypeBinding(mlir::StringRef, mlir::Location, const TypeBinding &, Flags, Frame = Frame());
  TypeBinding(
      mlir::StringRef Name, mlir::Location Loc, const TypeBinding &SuperType,
      ParamsMap GenericParams, Frame = Frame(), bool IsBuiltin = false
  );
  TypeBinding(
      mlir::StringRef, mlir::Location, const TypeBinding &SuperType, ParamsMap GenericParams, Flags,
      Frame = Frame()
  );
  TypeBinding(
      mlir::StringRef, mlir::Location, const TypeBinding &SuperType, ParamsMap GenericParams,
      ParamsMap ConstructorParams, MembersMap Members, Frame = Frame(), bool IsBuiltin = false
  );
  TypeBinding(
      mlir::StringRef, mlir::Location Loc, const TypeBinding &SuperType, ParamsMap GenericParams,
      ParamsMap ConstructorParams, MembersMap Members, Flags, Frame = Frame()
  );

  /// Constructs a type binding with a constant Val associated.
  TypeBinding(
      uint64_t Value, mlir::Location Loc, const TypeBindings &Bindings, bool IsBuiltin = false
  );
  TypeBinding(uint64_t Value, mlir::Location Loc, const TypeBindings &Bindings, Flags);

  ~TypeBinding() = default;

  //==---------------------------------------------------------------------==//
  // Factory methods
  //==---------------------------------------------------------------------==//

  /// Returns a copy of the type binding with the location replaced.
  static TypeBinding WithUpdatedLocation(const TypeBinding &t, mlir::Location loc) {
    TypeBinding copy = t;
    copy.loc = loc;
    return copy;
  }

  /// Returns a copy of the type binding with the frame replaced.
  static TypeBinding ReplaceFrame(const TypeBinding &t, Frame frame) {
    auto copy = t;
    copy.frame = frame;
    return copy;
  }

  /// Returns a copy of the type binding that has the variadic property set to true.
  static TypeBinding WrapVariadic(const TypeBinding &t) {
    TypeBinding copy = t;
    copy.flags.setVariadic();
    return copy;
  }

  /// Returns a new type binding that represents a generic parameter whose super type is the given
  /// type binding.
  static TypeBinding MakeGenericParam(const TypeBinding &t, llvm::StringRef name) {
    TypeBinding copy(name, t.loc, t);
    copy.flags.setGenericParamName();
    return copy;
  }

  /// Returns a copy of the type binding with the given constant expression associated.
  static TypeBinding WithExpr(const TypeBinding &b, expr::ConstExpr constExpr) {
    auto copy = b;
    copy.constExpr = constExpr;
    return copy;
  }

  /// Returns a copy of the type binding with any constant expression it may have removed.
  static TypeBinding NoExpr(const TypeBinding &b) {
    auto copy = b;
    copy.constExpr = expr::ConstExpr();
    return copy;
  }

  /// If the type binding is a constant value returns it super type, otherwise returns the given
  /// type binding.
  static const TypeBinding &StripConst(const TypeBinding &binding) {
    if (binding.isConst()) {
      return binding.getSuperType();
    }
    return binding;
  }

  /// Returns a copy of the type binding with the closure property set to true.
  static TypeBinding WithClosure(const TypeBinding &binding) {
    auto copy = binding;
    copy.flags.setClosure(true);
    return copy;
  }

  /// Returns a copy of the type binding with the closure property set to false.
  static TypeBinding WithoutClosure(const TypeBinding &binding) {
    auto copy = binding;
    copy.flags.setClosure(false);
    return copy;
  }

  //==---------------------------------------------------------------------==//
  // Utility methods & friends
  //==---------------------------------------------------------------------==//

  /// Prints a description of the type binding into the output stream.
  /// If fullPrintout is false prints a short description.
  /// If fullPrintout is true prints a detailed description.
  void print(llvm::raw_ostream &os, bool fullPrintout = false) const;

  /// Sets the type binding to take itself as a sigle constructor parameter.
  void selfConstructs();

  /// Marks the type binding as having been specialized.
  void markAsSpecialized() {
    assert(isGeneric());
    flags.setSpecialized(true);
  }

  /// Attempts to create an specialized version of the type using the provided parameters. Returns
  /// failure if the specialization fails.
  mlir::FailureOr<TypeBinding>
  specialize(EmitErrorFn emitError, mlir::ArrayRef<TypeBinding> params, TypeBindings &) const;

  bool operator==(const TypeBinding &) const;

  friend TypeBindings;

private:
  /// Locates the member in the inheritance chain. A component lower in the chain will shadow
  /// members in components higher in the chain. Returns failure if it was not found. If it was
  /// found but it couldn't get typechecked returns success wrapping a nullopt.
  mlir::FailureOr<std::optional<TypeBinding>> locateMember(mlir::StringRef) const;

  Flags flags;
  Name name;
  mlir::Location loc;
  expr::ConstExpr constExpr;
  const TypeBinding *superType;
  MembersMapPtr members;
  ParamsStoragePtr genericParams;
  ParamsStoragePtr constructorParams;
  Frame frame;
  FrameSlot *slot = nullptr;
};

mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, const TypeBinding &b);
mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, const TypeBinding::Name &name);

} // namespace zhl

namespace llvm {

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const zhl::TypeBinding &b);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const zhl::TypeBinding::Name &name);

} // namespace llvm

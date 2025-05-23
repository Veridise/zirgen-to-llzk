//===- Scope.h - Scope support during typing --------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a pure interface that defines a scope in the zirgen DSL
// and implementations of that interface for the different kinds of scopes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

namespace zhl {

class Scope {
public:
  enum ScopeKind {
    Sco_Component,
    Sco_Child,
    Sco_Frame,
    Sco_Block,
    Sco_ChildEnd,
  };
  explicit Scope(ScopeKind Kind) : kind(Kind) {}
  virtual ~Scope() = default;

  mlir::LogicalResult declareGlobal(mlir::StringRef, TypeBinding, EmitErrorFn);
  virtual void declareGenericParam(mlir::StringRef, uint64_t, TypeBinding) = 0;
  /// Declares a generic param as a lifted parameter that will materialize to an affine map.
  /// Returns a type binding pointing to the generic parameter name of the newly created parameter.
  virtual TypeBinding declareLiftedAffineToGenericParam(const TypeBinding &) = 0;
  virtual void declareConstructorParam(mlir::StringRef, uint64_t, TypeBinding) = 0;
  virtual void declareMember(mlir::StringRef) = 0;
  virtual void declareMember(mlir::StringRef, TypeBinding) = 0;
  virtual size_t memberCount() const = 0;
  virtual bool memberDeclaredWithType(mlir::StringRef) = 0;
  virtual void declareSuperType(TypeBinding) = 0;
  virtual zirgen::Zhl::ComponentOp getOp() const = 0;
  virtual mlir::FailureOr<TypeBinding> getSuperType() const = 0;
  virtual TypeBinding createBinding(mlir::StringRef name, mlir::Location loc) const = 0;
  virtual Frame &getCurrentFrame() = 0;
  virtual const Frame &getCurrentFrame() const = 0;
  virtual void setIsExtern() = 0;
  virtual void setNeedsBackVariablesSupport() = 0;

  ScopeKind getKind() const { return kind; }

private:
  ScopeKind kind;

  static MembersMap globals;
};

/// Mixin that implements the logic for defining members
class LexicalScopeImpl {
protected:
  void declareMemberImpl(mlir::StringRef);
  void declareMemberImpl(mlir::StringRef, TypeBinding);
  bool memberDeclaredWithTypeImpl(mlir::StringRef);
  MembersMap members;
};

class ComponentScope : public Scope, LexicalScopeImpl {
public:
  ComponentScope(zirgen::Zhl::ComponentOp component, TypeBindings &bindings);
  ~ComponentScope() override;

  // TODO: Add checks for duplicated names of parameters

  void declareGenericParam(mlir::StringRef name, uint64_t index, TypeBinding type) override;

  TypeBinding declareLiftedAffineToGenericParam(const TypeBinding &) override;

  void declareConstructorParam(mlir::StringRef name, uint64_t index, TypeBinding type) override;

  void declareMember(mlir::StringRef name) override;

  void declareSuperType(TypeBinding type) override;

  /// Allows overriding a member if the current value is None
  void declareMember(mlir::StringRef name, TypeBinding type) override;

  size_t memberCount() const override;

  bool memberDeclaredWithType(mlir::StringRef name) override;

  zirgen::Zhl::ComponentOp getOp() const override;

  mlir::FailureOr<TypeBinding> getSuperType() const override;

  TypeBinding createBinding(mlir::StringRef name, mlir::Location loc) const override;

  Frame &getCurrentFrame() override;
  const Frame &getCurrentFrame() const override;

  void setIsExtern() override { extern_ = true; }

  void setNeedsBackVariablesSupport() override { needsBVs = true; }

  static bool classof(const Scope *sco) { return sco->getKind() == Sco_Component; }

private:
  TypeBindings *bindings;
  zirgen::Zhl::ComponentOp component;
  ParamsMap constructorParams;
  ParamsMap genericParams, liftedParams;
  Frame frame;
  mlir::FailureOr<TypeBinding> superType;

  bool extern_ = false, needsBVs = false;
};

/// A scope that exists inside another
class ChildScope : public Scope {
public:
  explicit ChildScope(Scope &);
  void declareGenericParam(mlir::StringRef, uint64_t, TypeBinding) override;
  TypeBinding declareLiftedAffineToGenericParam(const TypeBinding &) override;
  void declareConstructorParam(mlir::StringRef, uint64_t, TypeBinding) override;
  void declareMember(mlir::StringRef) override;
  void declareMember(mlir::StringRef, TypeBinding) override;
  bool memberDeclaredWithType(mlir::StringRef) override;
  size_t memberCount() const override;
  void declareSuperType(TypeBinding) override;
  zirgen::Zhl::ComponentOp getOp() const override;
  mlir::FailureOr<TypeBinding> getSuperType() const override;
  Frame &getCurrentFrame() override;
  const Frame &getCurrentFrame() const override;
  TypeBinding createBinding(mlir::StringRef name, mlir::Location loc) const override;
  void setIsExtern() override;
  void setNeedsBackVariablesSupport() override;

  static bool classof(const Scope *sco) {
    return sco->getKind() >= Sco_Child && sco->getKind() < Sco_ChildEnd;
  }

protected:
  ChildScope(ScopeKind, Scope &);

private:
  Scope *parent;
};

/// A scope that allocates a new memory frame
class FrameScope : public ChildScope {
public:
  explicit FrameScope(Scope &, Frame);

  Frame &getCurrentFrame() override;
  const Frame &getCurrentFrame() const override;

  static bool classof(const Scope *sco) { return sco->getKind() == Sco_Frame; }

private:
  Frame frame;
};

/// Enters a new lexical scope where members can also be defined.
/// It has its own super type.
class BlockScope : public ChildScope, LexicalScopeImpl {
public:
  BlockScope(Scope &, TypeBindings &);

  void declareMember(mlir::StringRef name) override;

  void declareSuperType(TypeBinding type) override;

  /// Allows overriding a member if the current value is None
  void declareMember(mlir::StringRef name, TypeBinding type) override;

  size_t memberCount() const override;

  bool memberDeclaredWithType(mlir::StringRef name) override;

  mlir::FailureOr<TypeBinding> getSuperType() const override;

  TypeBinding createBinding(mlir::StringRef name, mlir::Location loc) const override;

  static bool classof(const Scope *sco) { return sco->getKind() == Sco_Block; }

private:
  TypeBindings *bindings;
  mlir::FailureOr<TypeBinding> superType;
};

} // namespace zhl

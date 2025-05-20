//===- Expr.h - Compile time expressions ------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes a collection of classes for handling compile time
// constant expressions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/simple_ilist.h>
#include <memory>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/Params.h>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
class Attribute;
class Builder;
} // namespace mlir

namespace zhl::expr {

using EmitErrorFn = llvm::function_ref<mlir::InFlightDiagnostic()>;

class ConstExpr;
class SimpleExprView;

namespace detail {

/// Root of the expression class hierarchy
class ExprBase : public llvm::ilist_node<ExprBase> {
public:
  ExprBase(const ExprBase &) = delete;
  ExprBase(ExprBase &&) = delete;
  ExprBase &operator=(const ExprBase &) = delete;
  ExprBase &operator=(ExprBase &&) = delete;
  virtual ~ExprBase() = default;

  enum ExprKind { Ex_Val, Ex_Ctor, Ex_Symbol };

  ExprKind getKind() const { return kind; }

  /// Creates a deep copy of this expression. The caller is responsible of handling the lifetime of
  /// the newly created object.
  virtual ExprBase *clone() const = 0;

  virtual bool operator==(const ExprBase &) const = 0;

  virtual void print(llvm::raw_ostream &) const = 0;

  /// Converts the expression into an MLIR Attribute
  virtual mlir::Attribute convertIntoAttribute(mlir::Builder &) const = 0;

  /// Attempts to convert the expression into an affine expression. Returns failure if it cannot do
  /// the conversion.
  virtual mlir::FailureOr<mlir::AffineExpr> convertIntoAffineExpr(mlir::Builder &) const = 0;

  /// Collects all the free symbols in the expression.
  virtual void collectFreeSymbols(llvm::StringSet<> &) const = 0;

  /// Remaps the free symbols in the expression to other expressions and returns
  /// the replaced expression. Matching TypeBindings must have a const expression.
  /// if that is not the case this method returns nullptr.
  virtual ExprBase *remap(Params, EmitErrorFn) const = 0;

  /// Computes a llvm compatible hash value
  virtual llvm::hash_code hash() const = 0;

  /// Wrap the expression into a simple view without lifetime considerations.
  operator SimpleExprView() const;

private:
  ExprKind kind;

protected:
  explicit ExprBase(ExprKind Kind) : kind(Kind) {}
};

llvm::hash_code hash_value(const ExprBase &);

} // namespace detail

/// Pure virtual class that exposes the common API of expressions around a safe wrapper that handles
/// when the view is not valid.
class ExprView {
public:
  virtual ~ExprView() = default;

  const detail::ExprBase &operator*() const { return ref(); }

  const detail::ExprBase *operator->() const { return get(); }

  /// Returns true if the view points to a valid object.
  operator bool() const { return get() != nullptr; }

  /// Clones the underlying expression and returns it wrapped in an adaptor of the same type.
  ConstExpr clone() const;

  bool operator==(const detail::ExprBase &other) const;

  bool operator==(const ExprView &other) const;

  /// Converts the expression into a MLIR Attribute
  mlir::Attribute convertIntoAttribute(mlir::Builder &builder) const;

  /// Attempts to converte the expresion into an affine expression. Returns failure if it failed to
  /// do so.
  mlir::FailureOr<mlir::AffineExpr> convertIntoAffineExpr(mlir::Builder &builder) const;

  /// Collects the free symbol names in the expression.
  void collectFreeSymbols(llvm::StringSet<> &FS) const;

  /// Convert the viewed expression into a managed expression. How this is achieved is
  /// implementation specific.
  virtual operator ConstExpr() const = 0;

  /// Returns a pointer to the viewed expression.
  virtual const detail::ExprBase *get() const = 0;

  /// Returns a reference to the viewed expression.
  virtual const detail::ExprBase &ref() const = 0;
};

/// Smart pointer around an expression.
class ConstExpr : public ExprView {
public:
  /// A view of a ConstExpr that leverages weak pointers for automatically invalidating the view
  /// if the underlying expression goes out of scope.
  class View : public ExprView {
  public:
    View(const std::shared_ptr<detail::ExprBase> &ptr) : view(ptr) {}

    /// Attempts to get a pointer to the value its weakly referencing to. Returns nullptr if its not
    /// valid.
    const detail::ExprBase *get() const override { return view.lock().get(); }

    /// Atemps to get a reference to the value its weakly referencing to. Aborts if the view is not
    /// valid.
    const detail::ExprBase &ref() const override { return *view.lock(); }

    friend ConstExpr;

    /// If the view is valid creates a ConstExpr that shares the pointer with other ConstExpr
    /// instances. If the view is not valid returns a falsey ConstExpr.
    operator ConstExpr() const override { return ConstExpr(*this); }

  private:
    std::weak_ptr<detail::ExprBase> view;
  };

  ConstExpr() = default;
  ConstExpr(std::nullptr_t) : ConstExpr() {}
  ConstExpr(const ConstExpr &) = default;
  ConstExpr &operator=(const ConstExpr &) = default;
  ConstExpr(ConstExpr &&) = default;
  ConstExpr &operator=(ConstExpr &&) = default;

  /// Constructs by copying the given expression.
  ConstExpr(const detail::ExprBase &Expr) : ConstExpr(Expr.clone()) {}

  /// Constructs by adopting the shared pointer the view is referencing to. If the view is invalid
  /// will construct a falsey object.
  ConstExpr(const View &view) : expr(view.view.lock()) {}

  /// Remaps the inner expression using the given parameters. If the replacement fails
  /// returns `mlir::failure()`.
  mlir::FailureOr<ConstExpr> remap(Params, EmitErrorFn) const;

  /// Returns a pointer to the underlying expression.
  const detail::ExprBase *get() const override { return expr.get(); }

  /// Returns a reference to the underlying expression.
  const detail::ExprBase &ref() const override { return *expr; }

  /// Returns a view pointing to the experssion managed by this object
  operator View() const { return View(expr); }

  /// Constructs a new ConstExpr holding a literal value.
  static ConstExpr Val(uint64_t);
  /// Constructs a new ConstExpr holding a reference to a symbol.
  static ConstExpr Symbol(mlir::StringRef, size_t);
  /// Constructs a new ConstExpr that constructs a new component passing ConstExprs as arguments. If
  /// any of the arguments is a falsey ConstExpr then this method returns a falsey ConstExpr.
  static ConstExpr Ctor(mlir::StringRef, mlir::ArrayRef<ConstExpr>);

  /// Trivially copies itself.
  operator ConstExpr() const override { return *this; }

private:
  explicit ConstExpr(detail::ExprBase *ptr) : expr(ptr) {}

  std::shared_ptr<detail::ExprBase> expr;
};

/// A simple view of an expression.
class SimpleExprView : public ExprView {
public:
  SimpleExprView(const detail::ExprBase &Arg) : arg(&Arg) {}

  /// Returns a pointer to the underlying expression.
  const detail::ExprBase *get() const override { return arg; }

  /// Returns a reference to the underlying expression.
  const detail::ExprBase &ref() const override {
    assert(arg);
    return *arg;
  }

  /// Clone the viewed expression and wrap it.
  operator ConstExpr() const override { return ConstExpr(*arg); }

private:
  const detail::ExprBase *arg;
};

namespace detail {

class Val : public ExprBase {
public:
  explicit Val(uint64_t Value) : ExprBase(Ex_Val), value(Value) {}

  static bool classof(const ExprBase *expr) { return expr->getKind() == Ex_Val; }

  ExprBase *clone() const override { return new Val(value); }

  bool operator==(const ExprBase &) const override;

  void print(llvm::raw_ostream &) const override;

  mlir::Attribute convertIntoAttribute(mlir::Builder &) const override;

  mlir::FailureOr<mlir::AffineExpr> convertIntoAffineExpr(mlir::Builder &) const override;

  void collectFreeSymbols(llvm::StringSet<> &) const override {}

  ExprBase *remap(Params, EmitErrorFn) const override;

  llvm::hash_code hash() const override;

  uint64_t getValue() const { return value; }

private:
  uint64_t value;
};

class Symbol : public ExprBase {
public:
  explicit Symbol(mlir::StringRef Name, size_t Pos) : ExprBase(Ex_Symbol), name(Name), pos(Pos) {}
  static bool classof(const ExprBase *expr) { return expr->getKind() == Ex_Symbol; }

  ExprBase *clone() const override { return new Symbol(name, pos); }

  bool operator==(const ExprBase &) const override;
  void print(llvm::raw_ostream &) const override;
  mlir::Attribute convertIntoAttribute(mlir::Builder &) const override;
  mlir::FailureOr<mlir::AffineExpr> convertIntoAffineExpr(mlir::Builder &) const override;

  void collectFreeSymbols(llvm::StringSet<> &symbols) const override { symbols.insert(name); }

  ExprBase *remap(Params, EmitErrorFn) const override;

  llvm::hash_code hash() const override;

  mlir::StringRef getName() const { return name; }

  size_t getPos() const { return pos; }

private:
  mlir::SmallString<5> name;
  size_t pos;
};

class Ctor : public ExprBase {
public:
  explicit Ctor(mlir::StringRef, mlir::ArrayRef<ConstExpr>);

  class Arguments {
    using ArgsList = llvm::simple_ilist<ExprBase>;

  public:
    using iterator = ArgsList::iterator;
    using const_iterator = ArgsList::const_iterator;
    using value_type = ExprBase *;
    using reference = ArgsList::reference;

    /// Clones the given expressions used as arguments.
    Arguments(mlir::SmallVectorImpl<value_type> &);
    /// Adopts ownership of the given expressions used as arguments. Cannot have null values in the
    /// vector.
    Arguments(mlir::SmallVectorImpl<value_type> &&);
    Arguments(const Arguments &);
    Arguments &operator=(const Arguments &);
    ~Arguments();

    iterator begin() { return lst.begin(); }
    const_iterator begin() const { return lst.begin(); }
    iterator end() { return lst.end(); }
    const_iterator end() const { return lst.end(); }

    size_t size() const { return lst.size(); }

    bool operator==(const Arguments &) const;

    ExprBase &operator[](size_t);
    const ExprBase &operator[](size_t) const;

  private:
    ArgsList lst;
  };

  static bool classof(const ExprBase *expr) { return expr->getKind() == Ex_Ctor; };

  ExprBase *clone() const override { return new Ctor(typeName, args); }

  bool operator==(const ExprBase &) const override;
  void print(llvm::raw_ostream &) const override;
  mlir::Attribute convertIntoAttribute(mlir::Builder &) const override;
  mlir::FailureOr<mlir::AffineExpr> convertIntoAffineExpr(mlir::Builder &) const override;
  void collectFreeSymbols(llvm::StringSet<> &) const override;

  ExprBase *remap(Params, EmitErrorFn) const override;

  llvm::hash_code hash() const override;

  Arguments &arguments() { return args; }
  const Arguments &arguments() const { return args; }

  mlir::StringRef getTypeName() const { return typeName; }

private:
  Ctor(mlir::StringRef Name, const Arguments &Args)
      : ExprBase(Ex_Ctor), args(Args), typeName(Name) {}

  Arguments args;
  mlir::SmallString<5> typeName;
};

} // namespace detail

/// An interface that coerces the underlying expression to a particular type.
/// An implementation of this interface needs to provide a pointer to the untyped expression.
template <typename Expr> class TypedExprView : public ExprView {
public:
  /// Returns a const pointer to the underlying expression.
  const Expr *get() const override { return mlir::cast_if_present<Expr>(base()); }

  /// Returns a const reference to the underlying expression
  const Expr &ref() const override {
    auto *E = get();
    assert(E);
    return *E;
  }

  /// Returns a reference to the expression of the given type. Aborts if the expression is not of
  /// that type.
  const Expr &operator*() const { return ref(); }

  /// Returns a pointer to the expression of the given type or nullptr if the underlying expression
  /// is not of that type.
  const Expr *operator->() const { return get(); }

  static bool classof(const ExprView *view) {
    return view && mlir::isa_and_present<Expr>(view->operator->());
  }

  /// Returns a pointer to the untyped expression.
  virtual const detail::ExprBase *base() const = 0;
};

/// A typed wrapper around a view that does not manage the lifetime of the view.
template <typename Expr> class TypedExprViewAdaptor : public TypedExprView<Expr> {
public:
  TypedExprViewAdaptor() : view(nullptr) {}
  TypedExprViewAdaptor(std::nullptr_t) : view(nullptr) {}
  TypedExprViewAdaptor(const ExprView &View) : view(&View) {}

  static bool classof(const ExprView *view) {
    return view && mlir::isa_and_present<Expr>(view->get());
  }

  const detail::ExprBase *base() const override {
    if (view) {
      return view->get();
    }
    return nullptr;
  }

  operator bool() const { return view && *view; }

  /// Delegates the convertion to ConstExpr to the inner view. Returns a falsey ConstExpr object if
  /// the view is null.
  operator ConstExpr() const override { return view ? ConstExpr(*view) : ConstExpr(); }

private:
  const ExprView *view;
};

using ValView = TypedExprViewAdaptor<detail::Val>;
using SymbolView = TypedExprViewAdaptor<detail::Symbol>;
using CtorView = TypedExprViewAdaptor<detail::Ctor>;

/// CRTP base class for expression adaptors that implements the common logic between them.
template <typename Expr> class TypedExprAdaptor : public TypedExprView<Expr> {
public:
  TypedExprAdaptor() : expr(nullptr) {}
  TypedExprAdaptor(std::nullptr_t) : expr(nullptr) {}
  TypedExprAdaptor(const TypedExprAdaptor &) = default;
  TypedExprAdaptor &operator=(const TypedExprAdaptor &) = default;
  TypedExprAdaptor(TypedExprAdaptor &&) = delete;
  TypedExprAdaptor &operator=(TypedExprAdaptor &&) = delete;

  /// Constructs an adaptor from an arbitrary expression. If the expression is not of the correct
  /// type this object constructs to a falsey object.
  TypedExprAdaptor(const ConstExpr &E) : expr(nullptr) {
    if (classof(&E)) {
      expr = E;
    }
  }

  /// Constructs an adaptor by cloning the given expression.
  TypedExprAdaptor(Expr &E) : expr(E) {}

  static bool classof(const ConstExpr *expr) {
    return expr && mlir::isa_and_present<Expr>(expr->get());
  }

  /// Returns a copy of the underlying ConstExpr object.
  operator ConstExpr() const override { return expr; }

  const detail::ExprBase *base() const override { return expr.get(); }

protected:
  ConstExpr expr;
};

/// Convenience adaptor for ConstExpr that holds a detail::Val
using ValExpr = TypedExprAdaptor<detail::Val>;

/// Convenience adaptor for ConstExpr that holds a detail::Symbol
using SymExpr = TypedExprAdaptor<detail::Symbol>;

/// Convenience adaptor for ConstExpr that holds a detail::Ctor
using CtorExpr = TypedExprAdaptor<detail::Ctor>;

llvm::hash_code hash_value(const ConstExpr &);

} // namespace zhl::expr

llvm::raw_ostream &operator<<(llvm::raw_ostream &, const zhl::expr::detail::ExprBase &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const zhl::expr::ExprView &);

#define enable_expr_cast(T, F)                                                                     \
  template <>                                                                                      \
  struct llvm::CastInfo<T, F> : public llvm::CastIsPossible<T, std::remove_const_t<F>> {           \
    static inline T doCast(const std::remove_const_t<F> &f) { return T(f); }                       \
    static inline T castFailed() { return T(); }                                                   \
    static inline T doCastIfPossible(const std::remove_const_t<F> &f) {                            \
      if (!llvm::CastInfo<T, F>::isPossible(f)) {                                                  \
        return castFailed();                                                                       \
      }                                                                                            \
      return doCast(f);                                                                            \
    }                                                                                              \
  }

enable_expr_cast(zhl::expr::ValExpr, zhl::expr::ConstExpr);
enable_expr_cast(zhl::expr::ValExpr, const zhl::expr::ConstExpr);
enable_expr_cast(zhl::expr::ValView, zhl::expr::ExprView);
enable_expr_cast(zhl::expr::ValView, const zhl::expr::ExprView);
enable_expr_cast(zhl::expr::SymExpr, zhl::expr::ConstExpr);
enable_expr_cast(zhl::expr::SymExpr, const zhl::expr::ConstExpr);
enable_expr_cast(zhl::expr::SymbolView, zhl::expr::ExprView);
enable_expr_cast(zhl::expr::SymbolView, const zhl::expr::ExprView);
enable_expr_cast(zhl::expr::CtorExpr, zhl::expr::ConstExpr);
enable_expr_cast(zhl::expr::CtorExpr, const zhl::expr::ConstExpr);
enable_expr_cast(zhl::expr::CtorView, zhl::expr::ExprView);
enable_expr_cast(zhl::expr::CtorView, const zhl::expr::ExprView);

#undef enable_expr_cast

#pragma once

#include <cstdint>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/simple_ilist.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/Support/LLVM.h>

namespace mlir {
class Attribute;
class Builder;
} // namespace mlir

namespace zhl::expr {

/// Root of the expression class hierarchy
class ExprBase : public llvm::ilist_node<ExprBase> {
public:
  ExprBase(const ExprBase &) = delete;
  ExprBase(ExprBase &&) = delete;
  ExprBase &operator=(const ExprBase &) = delete;
  ExprBase &operator=(ExprBase &&) = delete;
  virtual ~ExprBase() = default;

  enum ExprKind { Ex_Val, Ex_Ctor, Ex_Symbol };

  ExprKind getKind() const;

  virtual ExprBase *clone() const = 0;
  virtual bool operator==(const ExprBase &) const = 0;
  virtual void print(llvm::raw_ostream &) const = 0;
  virtual mlir::Attribute convertIntoAttribute(mlir::Builder &) const = 0;
  virtual mlir::FailureOr<mlir::AffineExpr> convertIntoAffineExpr(mlir::Builder &) const = 0;
  virtual void collectFreeSymbols(llvm::StringSet<> &) const = 0;

private:
  ExprKind kind;

protected:
  explicit ExprBase(ExprKind);
};

/// Smart pointer around an expression
class ConstExpr {
public:
  ConstExpr();
  ConstExpr(const ExprBase &);

  ExprBase *get();
  const ExprBase *get() const;
  ExprBase *operator->();
  const ExprBase *operator->() const;
  ExprBase &operator*();
  const ExprBase &operator*() const;

  static ConstExpr Val(uint64_t);
  static ConstExpr Symbol(mlir::StringRef, size_t);
  static ConstExpr Ctor(mlir::StringRef, mlir::ArrayRef<ConstExpr>);

  /// Returns true if the inner pointer points to a valid expression, false otherwise.
  operator bool() const;

  bool operator==(const ConstExpr &) const;

  mlir::Attribute convertIntoAttribute(mlir::Builder &) const;

private:
  explicit ConstExpr(ExprBase *);

  std::shared_ptr<ExprBase> expr;
};

namespace detail {

class Val : public ExprBase {
public:
  explicit Val(uint64_t);
  static bool classof(const ExprBase *);

  ExprBase *clone() const override;
  bool operator==(const ExprBase &) const override;
  void print(llvm::raw_ostream &) const override;
  mlir::Attribute convertIntoAttribute(mlir::Builder &) const override;
  mlir::FailureOr<mlir::AffineExpr> convertIntoAffineExpr(mlir::Builder &) const override;
  void collectFreeSymbols(llvm::StringSet<> &) const override;

  uint64_t getValue() const;

private:
  uint64_t value;
};

class Symbol : public ExprBase {
public:
  explicit Symbol(mlir::StringRef, size_t);
  static bool classof(const ExprBase *);

  ExprBase *clone() const override;
  bool operator==(const ExprBase &) const override;
  void print(llvm::raw_ostream &) const override;
  mlir::Attribute convertIntoAttribute(mlir::Builder &) const override;
  mlir::FailureOr<mlir::AffineExpr> convertIntoAffineExpr(mlir::Builder &) const override;
  void collectFreeSymbols(llvm::StringSet<> &) const override;

  mlir::StringRef getName() const;
  size_t getPos() const;

private:
  mlir::SmallString<5> name;
  size_t pos;
};

class Ctor : public ExprBase {
public:
  explicit Ctor(mlir::StringRef);

  class Arguments {
    using ArgsList = llvm::simple_ilist<ExprBase>;

  public:
    using iterator = ArgsList::iterator;
    using const_iterator = ArgsList::const_iterator;
    using value_type = ExprBase *;

    Arguments();
    Arguments(const Arguments &);
    Arguments &operator=(const Arguments &);
    ~Arguments();

    iterator begin();
    const_iterator begin() const;
    iterator end();
    const_iterator end() const;

    void push_back(ExprBase *);
    size_t size() const;

    bool operator==(const Arguments &) const;

    ExprBase &operator[](size_t);
    const ExprBase &operator[](size_t) const;

  private:
    void cleanArgsList();
    void copyArgsList(const Arguments &);

    ArgsList lst;
  };

  static bool classof(const ExprBase *);

  ExprBase *clone() const override;
  bool operator==(const ExprBase &) const override;
  void print(llvm::raw_ostream &) const override;
  mlir::Attribute convertIntoAttribute(mlir::Builder &) const override;
  mlir::FailureOr<mlir::AffineExpr> convertIntoAffineExpr(mlir::Builder &) const override;
  void collectFreeSymbols(llvm::StringSet<> &) const override;

  Arguments &arguments();
  const Arguments &arguments() const;

  mlir::StringRef getTypeName() const;

private:
  Ctor(mlir::StringRef, const Arguments &);

  Arguments args;
  mlir::SmallString<5> typeName;
};

} // namespace detail

/// Convenience adaptor for ConstExpr that holds a detail::Val
class ValExpr {
public:
  ValExpr();
  ValExpr(std::nullptr_t);
  ValExpr(const ValExpr &);
  ValExpr &operator=(const ValExpr &);
  ValExpr(ValExpr &&) = delete;
  ValExpr &operator=(ValExpr &&) = delete;
  ValExpr(const ConstExpr &);

  static bool classof(const ConstExpr *);

  uint64_t getValue() const;
  detail::Val &operator*();
  const detail::Val &operator*() const;
  /// Returns true if the inner pointer points to a valid expression, false otherwise.
  operator bool() const;

private:
  ConstExpr expr;
};

/// Convenience adaptor for ConstExpr that holds a detail::Sym
class SymExpr {
public:
  SymExpr();
  SymExpr(std::nullptr_t);
  SymExpr(const ConstExpr &);

  static bool classof(const ConstExpr *);
  detail::Symbol &operator*();
  const detail::Symbol &operator*() const;
  mlir::StringRef getName() const;
  size_t getPos() const;

  /// Returns true if the inner pointer points to a valid expression, false otherwise.
  operator bool() const;

private:
  ConstExpr expr;
};

/// Convenience adaptor for ConstExpr that holds a detail::Ctor
class CtorExpr {
public:
  using Arguments = detail::Ctor::Arguments;

  CtorExpr();
  CtorExpr(std::nullptr_t);
  CtorExpr(const ConstExpr &);

  detail::Ctor &operator*();
  const detail::Ctor &operator*() const;
  static bool classof(const ConstExpr *);
  Arguments &arguments();
  const Arguments &arguments() const;

  mlir::StringRef getTypeName() const;

  /// Returns true if the inner pointer points to a valid expression, false otherwise.
  operator bool() const;

private:
  ConstExpr expr;
};

} // namespace zhl::expr

llvm::raw_ostream &operator<<(llvm::raw_ostream &, const zhl::expr::ConstExpr &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const zhl::expr::detail::Val &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const zhl::expr::detail::Symbol &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const zhl::expr::detail::Ctor &);

template <>
struct llvm::CastInfo<zhl::expr::ValExpr, zhl::expr::ConstExpr>
    : public llvm::CastIsPossible<zhl::expr::ValExpr, zhl::expr::ConstExpr> {
  using from = zhl::expr::ConstExpr;
  using to = zhl::expr::ValExpr;
  using self = llvm::CastInfo<to, from>;

  static inline to doCast(const from &a) { return to(a); }
  static inline to castFailed() { return to(); }
  static inline to doCastIfPossible(const from &b) {
    if (!self::isPossible(b)) {
      return castFailed();
    }
    return doCast(b);
  }
};

template <>
struct llvm::CastInfo<zhl::expr::SymExpr, zhl::expr::ConstExpr>
    : public llvm::CastIsPossible<zhl::expr::SymExpr, zhl::expr::ConstExpr> {
  using from = zhl::expr::ConstExpr;
  using to = zhl::expr::SymExpr;
  using self = llvm::CastInfo<to, from>;

  static inline to doCast(const from &a) { return to(a); }
  static inline to castFailed() { return to(); }
  static inline to doCastIfPossible(const from &b) {
    if (!self::isPossible(b)) {
      return castFailed();
    }
    return doCast(b);
  }
};

template <>
struct llvm::CastInfo<zhl::expr::CtorExpr, zhl::expr::ConstExpr>
    : public llvm::CastIsPossible<zhl::expr::CtorExpr, zhl::expr::ConstExpr> {
  using from = zhl::expr::ConstExpr;
  using to = zhl::expr::CtorExpr;
  using self = llvm::CastInfo<to, from>;

  static inline to doCast(const from &a) { return to(a); }
  static inline to castFailed() { return to(); }
  static inline to doCastIfPossible(const from &b) {
    if (!self::isPossible(b)) {
      return castFailed();
    }
    return doCast(b);
  }
};

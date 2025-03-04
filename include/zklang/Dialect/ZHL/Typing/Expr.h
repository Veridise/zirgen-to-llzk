#pragma once

#include <cstdint>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/simple_ilist.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LLVM.h>

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

private:
  ExprKind kind;

protected:
  explicit ExprBase(ExprKind);
};

/// Smart pointer around an expression
class ConstExpr {
public:
  ConstExpr();

  ExprBase *get();
  const ExprBase *get() const;
  ExprBase *operator->();
  const ExprBase *operator->() const;
  ExprBase &operator*();
  const ExprBase &operator*() const;

  static ConstExpr Val(uint64_t);
  static ConstExpr Symbol(mlir::StringRef);
  static ConstExpr Ctor(mlir::StringRef, mlir::ArrayRef<ConstExpr>);

  /// Returns true if the inner pointer points to a valid expression, false otherwise.
  operator bool() const;

  bool operator==(const ConstExpr &) const;

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

  uint64_t getValue() const;

private:
  uint64_t value;
};

class Symbol : public ExprBase {
public:
  explicit Symbol(mlir::StringRef);
  static bool classof(const ExprBase *);

  ExprBase *clone() const override;
  bool operator==(const ExprBase &) const override;
  void print(llvm::raw_ostream &) const override;

  mlir::StringRef getName() const;

private:
  mlir::SmallString<5> name;
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
  explicit ValExpr(const ConstExpr &);

  static bool classof(const ConstExpr *);

  uint64_t getValue() const;

  /// Returns true if the inner pointer points to a valid expression, false otherwise.
  operator bool() const;

private:
  ConstExpr expr;
};

/// Convenience adaptor for ConstExpr that holds a detail::Sym
class SymExpr {
public:
  SymExpr();
  explicit SymExpr(const ConstExpr &);

  static bool classof(const ConstExpr *);

  mlir::StringRef getName() const;

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
  explicit CtorExpr(const ConstExpr &);

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

template <>
struct llvm::CastInfo<zhl::expr::ValExpr, zhl::expr::ConstExpr>
    : public llvm::CastIsPossible<zhl::expr::ValExpr, zhl::expr::ConstExpr> {
  static inline zhl::expr::ValExpr doCast(const zhl::expr::ConstExpr &a) {
    return zhl::expr::ValExpr(a);
  }
  static inline zhl::expr::ValExpr castFailed() { return zhl::expr::ValExpr(); }
  static inline zhl::expr::ValExpr doCastIfPossible(const zhl::expr::ConstExpr &b) {
    if (!CastInfo<zhl::expr::ValExpr, zhl::expr::ConstExpr>::isPossible(b)) {
      return castFailed();
    }
    return doCast(b);
  }
};

template <>
struct llvm::CastInfo<zhl::expr::SymExpr, zhl::expr::ConstExpr>
    : public llvm::CastIsPossible<zhl::expr::SymExpr, zhl::expr::ConstExpr> {
  static inline zhl::expr::SymExpr doCast(const zhl::expr::ConstExpr &a) {
    return zhl::expr::SymExpr(a);
  }
  static inline zhl::expr::SymExpr castFailed() { return zhl::expr::SymExpr(); }
  static inline zhl::expr::SymExpr doCastIfPossible(const zhl::expr::ConstExpr &b) {
    if (!CastInfo<zhl::expr::SymExpr, zhl::expr::ConstExpr>::isPossible(b)) {
      return castFailed();
    }
    return doCast(b);
  }
};

template <>
struct llvm::CastInfo<zhl::expr::CtorExpr, zhl::expr::ConstExpr>
    : public llvm::CastIsPossible<zhl::expr::CtorExpr, zhl::expr::ConstExpr> {
  static inline zhl::expr::CtorExpr doCast(const zhl::expr::ConstExpr &a) {
    return zhl::expr::CtorExpr(a);
  }
  static inline zhl::expr::CtorExpr castFailed() { return zhl::expr::CtorExpr(); }
  static inline zhl::expr::CtorExpr doCastIfPossible(const zhl::expr::ConstExpr &b) {
    if (!CastInfo<zhl::expr::CtorExpr, zhl::expr::ConstExpr>::isPossible(b)) {
      return castFailed();
    }
    return doCast(b);
  }
};

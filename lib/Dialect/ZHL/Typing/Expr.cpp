#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <memory>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/Expr.h>

using namespace mlir;

namespace zhl::expr {

//==-----------------------------------------------------------------------==//
// ExprBase
//==-----------------------------------------------------------------------==//

ExprBase::ExprBase(ExprKind Kind) : kind(Kind) {}

ExprBase::ExprKind ExprBase::getKind() const { return kind; }

//==-----------------------------------------------------------------------==//
// ConstExpr
//==-----------------------------------------------------------------------==//

ConstExpr::ConstExpr() : expr(nullptr) {}

ConstExpr::ConstExpr(ExprBase *exprPtr) : expr(exprPtr) {}

ConstExpr::operator bool() const { return expr != nullptr; }

bool ConstExpr::operator==(const ConstExpr &other) const {
  if (expr == nullptr && other.expr == nullptr) {
    return true;
  }
  if (expr == nullptr || other.expr == nullptr) {
    return false;
  }
  return *expr == *other.expr;
}

ExprBase *ConstExpr::get() { return expr.get(); }

const ExprBase *ConstExpr::get() const { return expr.get(); }

ExprBase *ConstExpr::operator->() { return expr.get(); }

const ExprBase *ConstExpr::operator->() const { return expr.get(); }

ExprBase &ConstExpr::operator*() { return *expr; }

const ExprBase &ConstExpr::operator*() const { return *expr; }

ConstExpr ConstExpr::Val(uint64_t value) { return ConstExpr(new detail::Val(value)); }

ConstExpr ConstExpr::Symbol(StringRef name, size_t pos) {
  return ConstExpr(new detail::Symbol(name, pos));
}

namespace {
template <typename T> inline bool asBool(const T &t) { return t; }
} // namespace

ConstExpr ConstExpr::Ctor(StringRef name, ArrayRef<ConstExpr> args) {
  if (std::all_of(args.begin(), args.end(), asBool<ConstExpr>)) {
    auto *expr = new detail::Ctor(name);
    std::transform(args.begin(), args.end(), std::back_inserter(expr->arguments()), [](auto CE) {
      return CE->clone();
    });
    return ConstExpr(expr);
  }
  return ConstExpr();
}

Attribute ConstExpr::convertIntoAttribute(Builder &builder) const {
  if (expr) {
    return expr->convertIntoAttribute(builder);
  }
  return nullptr;
}

//==-----------------------------------------------------------------------==//
// ValExpr
//==-----------------------------------------------------------------------==//

ValExpr::ValExpr() = default;

ValExpr::ValExpr(const ConstExpr &Expr) : expr(Expr) {}

bool ValExpr::classof(const ConstExpr *expr) {
  if (!(expr && expr->get())) {
    return false;
  }
  return mlir::isa<detail::Val>(**expr);
}

uint64_t ValExpr::getValue() const { return mlir::cast<detail::Val>(expr.get())->getValue(); }

ValExpr::operator bool() const { return expr; }

//==-----------------------------------------------------------------------==//
// SymExpr
//==-----------------------------------------------------------------------==//

SymExpr::SymExpr() = default;

SymExpr::SymExpr(const ConstExpr &Expr) : expr(Expr) {}

bool SymExpr::classof(const ConstExpr *expr) {
  if (!(expr && expr->get())) {
    return false;
  }
  return mlir::isa<detail::Symbol>(**expr);
}

StringRef SymExpr::getName() const { return mlir::cast<detail::Symbol>(expr.get())->getName(); }

SymExpr::operator bool() const { return expr; }

//==-----------------------------------------------------------------------==//
// CtorExpr
//==-----------------------------------------------------------------------==//

CtorExpr::CtorExpr() = default;

CtorExpr::CtorExpr(const ConstExpr &Expr) : expr(Expr) {}

bool CtorExpr::classof(const ConstExpr *expr) {
  if (!(expr && expr->get())) {
    return false;
  }
  return mlir::isa<detail::Ctor>(**expr);
}

CtorExpr::Arguments &CtorExpr::arguments() {
  return mlir::dyn_cast<detail::Ctor>(expr.get())->arguments();
}

const CtorExpr::Arguments &CtorExpr::arguments() const {
  return mlir::dyn_cast<detail::Ctor>(expr.get())->arguments();
}

StringRef CtorExpr::getTypeName() const {
  return mlir::cast<detail::Ctor>(expr.get())->getTypeName();
}

CtorExpr::operator bool() const { return expr; }

namespace detail {

//==-----------------------------------------------------------------------==//
// Val
//==-----------------------------------------------------------------------==//

Val::Val(uint64_t Value) : ExprBase(Ex_Val), value(Value) {}

bool Val::classof(const ExprBase *expr) { return expr->getKind() == Ex_Val; }

ExprBase *Val::clone() const { return new Val(value); }

bool Val::operator==(const ExprBase &other) const {
  if (auto *otherVal = mlir::dyn_cast<Val>(&other)) {
    return value == otherVal->value;
  }
  return false;
}

void Val::print(llvm::raw_ostream &os) const { os << "Val(" << value << ")"; }

uint64_t Val::getValue() const { return value; }

Attribute Val::convertIntoAttribute(Builder &builder) const {
  return builder.getIntegerAttr(builder.getI64Type(), value);
}

//==-----------------------------------------------------------------------==//
// Symbol
//==-----------------------------------------------------------------------==//

Symbol::Symbol(StringRef Name, size_t Pos) : ExprBase(Ex_Symbol), name(Name), pos(Pos) {}

bool Symbol::classof(const ExprBase *expr) { return expr->getKind() == Ex_Symbol; }

ExprBase *Symbol::clone() const { return new Symbol(name, pos); }

bool Symbol::operator==(const ExprBase &other) const {
  if (auto *otherSym = mlir::dyn_cast<Symbol>(&other)) {
    return name == otherSym->name;
  }
  return false;
}

void Symbol::print(llvm::raw_ostream &os) const { os << "Sym(" << name << ")"; }

StringRef Symbol::getName() const { return name; }

size_t Symbol::getPos() const { return pos; }

Attribute Symbol::convertIntoAttribute(Builder &builder) const {
  return SymbolRefAttr::get(builder.getStringAttr(name));
}

//==-----------------------------------------------------------------------==//
// Ctor
//==-----------------------------------------------------------------------==//

Ctor::Ctor(StringRef Name) : ExprBase(Ex_Ctor), typeName(Name) {}

Ctor::Ctor(StringRef Name, const Arguments &Args) : ExprBase(Ex_Ctor), args(Args), typeName(Name) {}

bool Ctor::classof(const ExprBase *expr) { return expr->getKind() == Ex_Ctor; }

ExprBase *Ctor::clone() const { return new Ctor(typeName, args); }

bool Ctor::operator==(const ExprBase &other) const {
  if (auto *otherCtor = mlir::dyn_cast<Ctor>(&other)) {
    return typeName == otherCtor->typeName && args == otherCtor->args;
  }
  return false;
}

namespace {

inline bool isInfixExprByName(StringRef name) {
  return llvm::StringSwitch<bool>(name)
      .Case("Add", true)
      .Case("Sub", true)
      .Case("Mul", true)
      .Case("Div", true)
      .Default(false);
}

inline StringRef getInfixSymbol(StringRef name) {
  return llvm::StringSwitch<StringRef>(name)
      .Case("Add", "+")
      .Case("Sub", "-")
      .Case("Mul", "*")
      .Case("Div", "/")
      .Default(name);
}

inline bool isInfixExpr(const ExprBase &expr) {
  if (auto ctor = mlir::dyn_cast<Ctor>(&expr)) {
    return ctor->arguments().size() == 2 && isInfixExprByName(ctor->getTypeName());
  }
  return false;
}

} // namespace

void Ctor::print(llvm::raw_ostream &os) const {
  if (isInfixExpr(*this)) {
    bool lhsNeedsParen = isInfixExpr(args[0]);
    if (lhsNeedsParen) {
      os << "(";
    }
    args[0].print(os);
    if (lhsNeedsParen) {
      os << ")";
    }
    os << " " << getInfixSymbol(typeName) << " ";
    bool rhsNeedsParen = isInfixExpr(args[1]);
    if (rhsNeedsParen) {
      os << "(";
    }
    args[1].print(os);
    if (rhsNeedsParen) {
      os << ")";
    }
  } else {
    os << typeName << "(";
    llvm::interleaveComma(args, os, [&](auto &e) { e.print(os); });
    os << ")";
  }
}

Ctor::Arguments &Ctor::arguments() { return args; }

const Ctor::Arguments &Ctor::arguments() const { return args; }

StringRef Ctor::getTypeName() const { return typeName; }

// Ctor::convertIntoAttribute implementation is in Interpreter.cpp because it needs information
// about semi-affine expressions

//==-----------------------------------------------------------------------==//
// Ctor::Arguments
//==-----------------------------------------------------------------------==//

Ctor::Arguments::Arguments() = default;

Ctor::Arguments::Arguments(const Arguments &other) { copyArgsList(other); }

Ctor::Arguments &Ctor::Arguments::operator=(const Arguments &other) {
  cleanArgsList();
  copyArgsList(other);
  return *this;
}

Ctor::Arguments::~Arguments() { cleanArgsList(); }

Ctor::Arguments::iterator Ctor::Arguments::begin() { return lst.begin(); }

Ctor::Arguments::const_iterator Ctor::Arguments::begin() const { return lst.begin(); }

Ctor::Arguments::iterator Ctor::Arguments::end() { return lst.end(); }

Ctor::Arguments::const_iterator Ctor::Arguments::end() const { return lst.end(); }

void Ctor::Arguments::push_back(ExprBase *expr) { lst.insert(lst.end(), *expr); }

void Ctor::Arguments::cleanArgsList() { lst.clearAndDispose(std::default_delete<ExprBase>()); }

void Ctor::Arguments::copyArgsList(const Arguments &other) {
  std::transform(other.begin(), other.end(), std::back_inserter(*this), [](auto &arg) {
    return arg.clone();
  });
}

bool Ctor::Arguments::operator==(const Arguments &other) const {
  if (lst.size() != other.lst.size()) {
    return false;
  }

  for (auto [lhs, rhs] : llvm::zip_equal(lst, other.lst)) {
    if (lhs != rhs) {
      return false;
    }
  }
  return true;
}

ExprBase &Ctor::Arguments::operator[](size_t offset) {
  assert(offset < lst.size());
  return *std::next(lst.begin(), offset);
}

const ExprBase &Ctor::Arguments::operator[](size_t offset) const {
  assert(offset < lst.size());
  return *std::next(lst.begin(), offset);
}

size_t Ctor::Arguments::size() const { return lst.size(); }

} // namespace detail

} // namespace zhl::expr

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const zhl::expr::ConstExpr &expr) {
  if (expr) {
    expr->print(os);
  } else {
    os << "ConstExpr(<<NULL>>)";
  }
  return os;
}

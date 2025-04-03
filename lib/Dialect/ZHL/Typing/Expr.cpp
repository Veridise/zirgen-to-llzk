//===- Expr.cpp - Constant exprs implementation -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/Expr.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Debug.h>
#include <memory>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZML/IR/Attrs.h>

#define DEBUG_TYPE "zhl-const-exprs"

using namespace mlir;
using namespace zhl::expr;
using namespace zhl::expr::detail;

//==-----------------------------------------------------------------------==//
// ExprBase
//==-----------------------------------------------------------------------==//

ExprBase::operator SimpleExprView() const { return SimpleExprView(*this); }

//==-----------------------------------------------------------------------==//
// ExprView
//==-----------------------------------------------------------------------==//

ConstExpr ExprView::clone() const {
  if (auto *expr = get()) {
    return ConstExpr(*expr);
  }
  return ConstExpr(nullptr);
}

bool ExprView::operator==(const detail::ExprBase &other) const {
  if (auto *expr = get()) {
    return *expr == other;
  }
  return false;
}

bool ExprView::operator==(const ExprView &other) const {
  auto *expr = get();
  if (expr && other) {
    return expr->operator==(other.ref());
  }
  return !expr && !other;
}

/// Converts the expression into a MLIR Attribute
Attribute ExprView::convertIntoAttribute(mlir::Builder &builder) const {
  if (auto *expr = get()) {
    return expr->convertIntoAttribute(builder);
  }
  return nullptr;
}

/// Attempts to converte the expresion into an affine expression. Returns failure if it failed to
/// do so.
FailureOr<AffineExpr> ExprView::convertIntoAffineExpr(mlir::Builder &builder) const {
  if (auto *expr = get()) {
    return expr->convertIntoAffineExpr(builder);
  }
  return mlir::failure();
}

/// Collects the free symbol names in the expression.
void ExprView::collectFreeSymbols(llvm::StringSet<> &FS) const {
  if (auto *expr = get()) {
    expr->collectFreeSymbols(FS);
  }
}

//==-----------------------------------------------------------------------==//
// ConstExpr
//==-----------------------------------------------------------------------==//

ConstExpr ConstExpr::Val(uint64_t value) { return ConstExpr(new class Val(value)); }

ConstExpr ConstExpr::Symbol(StringRef name, size_t pos) {
  return ConstExpr(new class Symbol(name, pos));
}

template <typename T> static bool asBool(const T &t) { return t; }

ConstExpr ConstExpr::Ctor(StringRef name, ArrayRef<ConstExpr> args) {
  if (llvm::all_of(args, asBool<ConstExpr>)) {
    return ConstExpr(new class Ctor(name, args));
  }
  return ConstExpr();
}

//==-----------------------------------------------------------------------==//
// Val
//==-----------------------------------------------------------------------==//

bool Val::operator==(const ExprBase &other) const {
  if (auto *otherVal = mlir::dyn_cast<Val>(&other)) {
    return value == otherVal->value;
  }
  return false;
}

void Val::print(llvm::raw_ostream &os) const { os << value; }

Attribute Val::convertIntoAttribute(Builder &builder) const {
  assert(value <= std::numeric_limits<int64_t>::max());
  return builder.getI64IntegerAttr(static_cast<int64_t>(value));
}

//==-----------------------------------------------------------------------==//
// Symbol
//==-----------------------------------------------------------------------==//

bool Symbol::operator==(const ExprBase &other) const {
  if (auto *otherSym = mlir::dyn_cast<Symbol>(&other)) {
    return name == otherSym->name;
  }
  return false;
}

void Symbol::print(llvm::raw_ostream &os) const { os << name; }

Attribute Symbol::convertIntoAttribute(Builder &builder) const {
  return SymbolRefAttr::get(builder.getStringAttr(name));
}

//==-----------------------------------------------------------------------==//
// Ctor
//==-----------------------------------------------------------------------==//

Ctor::Ctor(StringRef Name, ArrayRef<ConstExpr> Args)
    : ExprBase(Ex_Ctor),
      args(llvm::map_to_vector(Args, [](auto &expr) -> ExprBase * { return expr->clone(); })),
      typeName(Name) {}

bool Ctor::operator==(const ExprBase &other) const {
  if (auto *otherCtor = mlir::dyn_cast<Ctor>(&other)) {
    return typeName == otherCtor->typeName && args == otherCtor->args;
  }
  return false;
}

static bool isInfixExprByName(StringRef name) {
  return llvm::StringSwitch<bool>(name)
      .Case("Add", true)
      .Case("Sub", true)
      .Case("Mul", true)
      .Case("Div", true)
      .Default(false);
}

static StringRef getInfixSymbol(StringRef name) {
  return llvm::StringSwitch<StringRef>(name)
      .Case("Add", "+")
      .Case("Sub", "-")
      .Case("Mul", "*")
      .Case("Div", "/")
      .Default(name);
}

static bool isInfixExpr(const ExprBase &expr) {
  if (auto ctor = mlir::dyn_cast<Ctor>(&expr)) {
    return ctor->arguments().size() == 2 && isInfixExprByName(ctor->getTypeName());
  }
  return false;
}

static void printSide(llvm::raw_ostream &os, const ExprBase &expr) {
  bool needsParen = isInfixExpr(expr);
  if (needsParen) {
    os << "(";
  }
  expr.print(os);
  if (needsParen) {
    os << ")";
  }
}

void Ctor::print(llvm::raw_ostream &os) const {
  if (isInfixExpr(*this)) {
    printSide(os, args[0]);
    os << " " << getInfixSymbol(typeName) << " ";
    printSide(os, args[1]);
  } else if (getTypeName() == "Neg") {
    os << "-";
    printSide(os, args[0]);
  } else {
    os << typeName << "(";
    llvm::interleaveComma(args, os, [&](auto &e) { e.print(os); });
    os << ")";
  }
}

void Ctor::collectFreeSymbols(llvm::StringSet<> &symbols) const {
  for (auto &arg : args) {
    arg.collectFreeSymbols(symbols);
  }
}

Attribute Ctor::convertIntoAttribute(Builder &builder) const {
  auto expr = convertIntoAffineExpr(builder);
  if (failed(expr)) {
    return nullptr;
  }
  LLVM_DEBUG(llvm::dbgs() << "Generated expression: " << expr << "\n");

  std::optional<unsigned int> largest;
  expr->walk([&](AffineExpr e) {
    if (!mlir::isa<AffineSymbolExpr>(e)) {
      return;
    }
    auto symExpr = mlir::cast<AffineSymbolExpr>(e);
    largest = std::max(largest.value_or(0), symExpr.getPosition());
  });
  unsigned int symbolCount = 0;
  if (largest.has_value()) {
    symbolCount = *largest + 1;
  }
  auto map = AffineMap::get(0, symbolCount, *expr);

  // After constructing the map reduce the number of symbols and compute what formals they refer to
  SmallVector<uint64_t> formals;
  SmallVector<AffineExpr> formalsToSymbols;
  unsigned int shift = 0;
  for (unsigned int i = 0; i < map.getNumSymbols(); i++) {
    if (map.isFunctionOfSymbol(i)) {
      formals.push_back(i);
      // Map the i-th formal to the position the symbol will have in the affine map after
      // removing dead symbols
      formalsToSymbols.push_back(builder.getAffineSymbolExpr(i - shift));
    } else {
      shift++;
      formalsToSymbols.push_back(builder.getAffineConstantExpr(0));
    }
  }
  assert(formalsToSymbols.size() == map.getNumSymbols());
  assert(formals.size() <= std::numeric_limits<unsigned>::max());
  map = map.replaceDimsAndSymbols({}, formalsToSymbols, 0, static_cast<unsigned>(formals.size()));
  return zml::ConstExprAttr::get(map, formals);
}

//==-----------------------------------------------------------------------==//
// Ctor::Arguments
//==-----------------------------------------------------------------------==//

static ExprBase &cloneHelper(const ExprBase &expr) { return *expr.clone(); }

template <typename In, typename Out, typename Fn>
static void copyArgsList(const In &in, Out &out, Fn f) {
  for (auto &arg : in) {
    out.push_back(f(arg));
  }
}

template <typename List> static void cleanArgsList(List &lst) {
  lst.clearAndDispose(std::default_delete<typename List::value_type>());
}

Ctor::Arguments::Arguments(SmallVectorImpl<value_type> &&Args) {
  copyArgsList(Args, lst, [](value_type expr) -> reference { return cloneHelper(*expr); });
}

Ctor::Arguments::Arguments(const Arguments &other) { copyArgsList(other.lst, lst, cloneHelper); }

Ctor::Arguments &Ctor::Arguments::operator=(const Arguments &other) {
  cleanArgsList(lst);
  copyArgsList(other.lst, lst, cloneHelper);
  return *this;
}

Ctor::Arguments::~Arguments() { cleanArgsList(lst); }

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
  assert(offset <= std::numeric_limits<ptrdiff_t>::max());
  return *std::next(lst.begin(), static_cast<ptrdiff_t>(offset));
}

const ExprBase &Ctor::Arguments::operator[](size_t offset) const {
  assert(offset < lst.size());
  assert(offset <= std::numeric_limits<ptrdiff_t>::max());
  return *std::next(lst.begin(), static_cast<ptrdiff_t>(offset));
}

//==-----------------------------------------------------------------------==//
// Stream overloads
//==-----------------------------------------------------------------------==//

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ExprView &expr) {
  if (expr) {
    expr->print(os);
  } else {
    os << "<<NULL>>";
  }
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ExprBase &expr) {
  expr.print(os);
  return os;
}

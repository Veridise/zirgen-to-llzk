//===- Params.h - Views over a set of parameters ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file includes view classes that allows handling a collection of
// parameters, supporting both generic parameters and constructor parameters.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace llvm {
class raw_ostream;
}

namespace zhl {

class TypeBinding;
struct ParamsStorage;
class ParamsMap;

using ParamName = std::string;
using ParamsList = mlir::SmallVector<TypeBinding, 0>;
using ParamNames = mlir::SmallVector<ParamName>;

/// Configuration for the print methods.
struct ParamsPrintCfg {
  bool fullPrintout = false;
  char printIfEmpty = false;
  char header = '<';
  char footer = '>';
};

class Params {
public:
  using iterator = mlir::ArrayRef<TypeBinding>::const_iterator;

  Params(const ParamsStorage &params) : sto(&params) {}

  /// Returns the number of parameters.
  size_t size() const;

  /// Returns the number of parameter that were declared in the source language.
  size_t sizeOfDeclared() const;

  /// Creates a copy of the storage in map form.
  operator ParamsMap() const;

  /// Returns the name of the parameter at the given index.
  mlir::StringRef getName(size_t) const;

  /// Returns the type binding of the parameter at the given index.
  const TypeBinding &getParam(size_t) const;

  /// Returns an array of the parameters names.
  mlir::ArrayRef<ParamName> getNames() const;

  /// Returns an array of the type binding of the parameters.
  mlir::ArrayRef<TypeBinding> getParams() const;

  /// Returns an array of bits that declare what parameters have been injected.
  mlir::SmallVector<bool> getInjectedStatus() const;

  /// Returns a vector of type bindings of the parameters that were declared in the source language.
  ParamsList getDeclaredParams() const;

  /// Returns a pointer to a parameter that matches the given name or nullptr if it was not found.
  const TypeBinding *operator[](mlir::StringRef name) const;

  /// Prints the parameters as a map from the names to the type bindings into the output stream.
  /// Uses PrintCfg::fullPrintout and ignores the rest of the passed configuration.
  void printMapping(llvm::raw_ostream &os, ParamsPrintCfg cfg = {}) const;

  /// Prints the names of the parameters into the output stream.
  /// Ignores PrintCfg::fullPrintout.
  void printNames(llvm::raw_ostream &os, ParamsPrintCfg cfg = {}) const;

  /// Prints the type bindings of the parameters into the output stream.
  void printParams(llvm::raw_ostream &os, ParamsPrintCfg cfg = {}) const;

  iterator begin() const;
  iterator end() const;

  /// Returns true if there is a parameter that matches the given name.
  bool contains(mlir::StringRef) const;

  /// Returns true if there are no parameters.
  bool empty() const;

  bool operator==(const Params &) const;

  const ParamsStorage *data() const { return sto; }

  /// Returns all the data related to each parameter in a zip sequence.
  auto zipped() const { return llvm::zip_equal(getParams(), getNames(), getInjectedStatus()); }

private:
  const ParamsStorage *sto;
};

class MutableParams : public Params {
public:
  using iterator = mlir::MutableArrayRef<TypeBinding>::iterator;

  MutableParams(ParamsStorage &params) : Params(params) {}

  /// If there is a parameter with the same given name assigns the given binding to it.
  void replaceParam(mlir::StringRef name, const TypeBinding &binding);

  /// Returns the parameter at the specified index.
  TypeBinding &getParam(size_t) const;

  /// Returns a mutable array of the parameters type bindings.
  mlir::MutableArrayRef<TypeBinding> getParams() const;

  /// Returns a pointer to a parameter that matches the given name or nullptr if it was not found.
  TypeBinding *operator[](mlir::StringRef name) const {
    auto ptr = Params::operator[](name);
    return const_cast<TypeBinding *>(ptr);
  }

  iterator begin() const;
  iterator end() const;

  ParamsStorage *data() const { return const_cast<ParamsStorage *>(Params::data()); }
};

} // namespace zhl

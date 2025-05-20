//===- Params.cpp - Views and storage over a set of parameters --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/Params.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

using namespace zhl;
using namespace mlir;

//==-----------------------------------------------------------------------==//
// ParamsStorage
//==-----------------------------------------------------------------------==//

ParamsStorage::ParamsStorage(
    const ParamsMap &map, size_t size, std::optional<TypeBinding> defaultBinding
)
    : names(size),
      injected((
          assert(size <= std::numeric_limits<unsigned int>::max()), static_cast<unsigned int>(size)
      )) {
  // Hack to get the bindings ordered without having a default constructor
  SmallVector<const TypeBinding *> tmp(size, nullptr);
  for (auto &entry : map) {
    const ParamData &val = entry.getValue();
    uint64_t pos = val.Pos;
    tmp[pos] = &val.Type;
    names[pos] = entry.getKey();
    assert(pos <= std::numeric_limits<unsigned int>::max());
    injected[static_cast<unsigned int>(pos)] = val.Injected;
  }
  params.reserve(size);
  for (const TypeBinding *type : tmp) {
    if (type) {
      params.push_back(*type);
    } else {
      assert(defaultBinding.has_value() && "cannot have null type when there is no default");
      params.push_back(defaultBinding.value());
    }
  }
}

//==-----------------------------------------------------------------------==//
// Params
//==-----------------------------------------------------------------------==//

Params::operator ParamsMap() const {
  ParamsMap map;
  for (size_t i = 0; i < data()->params.size(); i++) {
    assert(i <= std::numeric_limits<unsigned int>::max());
    bool injected = data()->injected[static_cast<unsigned int>(i)];
    map.declare(data()->names[i], data()->params[i], i, injected);
  }
  return map;
}

const TypeBinding &Params::getParam(size_t i) const {
  assert(i < data()->params.size());
  return data()->params[i];
}

ArrayRef<ParamName> Params::getNames() const { return data()->names; }

SmallVector<bool> Params::getInjectedStatus() const {
  return llvm::map_to_vector(
      llvm::index_range(0, data()->injected.size()),
      [this](unsigned int Idx) -> bool { return data()->injected[Idx]; }
  );
}

StringRef Params::getName(size_t i) const {
  assert(i < data()->names.size());
  return data()->names[i];
}

size_t Params::size() const { return data()->params.size(); }

size_t Params::sizeOfDeclared() const { return data()->params.size() - data()->injected.count(); }

mlir::ArrayRef<TypeBinding> Params::getParams() const { return data()->params; }

ParamsList Params::getDeclaredParams() const {
  ParamsList params;

  for (size_t i = 0; i < data()->params.size(); i++) {
    assert(i <= std::numeric_limits<unsigned int>::max());
    if (!data()->injected[static_cast<unsigned int>(i)]) {
      params.push_back(data()->params[i]);
    }
  }

  return params;
}

const TypeBinding *Params::operator[](StringRef name) const {
  for (size_t i = 0; i < data()->names.size(); i++) {
    if (data()->names[i] == name) {
      return &data()->params[i];
    }
  }
  return nullptr;
}

template <typename It, typename UnaryFn, typename Side>
static void printHelper(It it, llvm::raw_ostream &os, UnaryFn fn, Side header, Side footer) {
  os << header;
  llvm::interleaveComma(it, os, fn);
  os << footer;
}

void Params::printMapping(llvm::raw_ostream &os, ParamsPrintCfg cfg) const {
  printHelper(llvm::zip_equal(data()->names, data()->params), os, [&](auto tup) {
    auto [name, param] = tup;
    os << name << ": ";
    param.print(os, cfg.fullPrintout);
  }, "{ ", " }");
}

void Params::printNames(llvm::raw_ostream &os, ParamsPrintCfg cfg) const {
  if (!empty() || cfg.printIfEmpty) {
    printHelper(data()->names, os, [&](const auto &e) { os << e; }, cfg.header, cfg.footer);
  }
}

void Params::printParams(llvm::raw_ostream &os, ParamsPrintCfg cfg) const {
  if (!empty() || cfg.printIfEmpty) {
    printHelper(data()->params, os, [&](const auto &e) {
      e.print(os, cfg.fullPrintout);
    }, cfg.header, cfg.footer);
  }
}

Params::iterator Params::begin() const { return data()->params.begin(); }

Params::iterator Params::end() const { return data()->params.end(); }

bool Params::contains(StringRef name) const {
  auto B = data()->names.begin();
  auto E = data()->names.end();
  auto pos = std::find(B, E, name);
  return pos != E;
}

bool Params::empty() const { return data()->names.empty(); }

bool Params::operator==(const Params &other) const {
  return data()->params == other.data()->params && data()->names == other.data()->names;
}

llvm::hash_code zhl::hash_value(const Params &params) {
  auto paramsHash =
      llvm::hash_combine_range(params.data()->params.begin(), params.data()->params.end());
  auto namesHash =
      llvm::hash_combine_range(params.data()->names.begin(), params.data()->names.end());
  // llvm::BitVector does not implement an iterator of bools over the whole set.
  SmallVector<bool> bits;
  auto &bitsSrc = params.data()->injected;
  auto size = bitsSrc.size();
  bits.reserve(size);
  for (unsigned i = 0; i < size; i++) {
    bits.push_back(bitsSrc[i]);
  }
  auto injectedHash = llvm::hash_combine_range(bits.begin(), bits.end());

  return llvm::hash_combine("Params", paramsHash, "names", namesHash, "injected", injectedHash);
}

//==-----------------------------------------------------------------------==//
// MutableParams
//==-----------------------------------------------------------------------==//

void MutableParams::replaceParam(StringRef name, const TypeBinding &binding) {
  auto found = this->operator[](name);
  if (found != nullptr) {
    *found = binding;
  }
}

TypeBinding &MutableParams::getParam(size_t i) const {
  assert(i < data()->params.size());
  return data()->params[i];
}

mlir::MutableArrayRef<TypeBinding> MutableParams::getParams() const { return data()->params; }

MutableParams::iterator MutableParams::begin() const { return data()->params.begin(); }

MutableParams::iterator MutableParams::end() const { return data()->params.end(); }

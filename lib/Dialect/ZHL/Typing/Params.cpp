#include <cassert>
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

static void fillVectors(
    const ParamsMap &map, SmallVectorImpl<const TypeBinding *> &tmp,
    SmallVectorImpl<ParamName> &names, BitVector &injected
) {
  for (auto &entry : map) {
    auto pos = entry.getValue().Pos;
    tmp[pos] = &entry.getValue().Type;
    names[pos] = entry.getKey();
    injected[pos] = entry.getValue().Injected;
  }
}

ParamsStorage::ParamsStorage(const ParamsMap &map) : names(map.size()), injected(map.size()) {
  // Hack to get the bindings ordered without having a default constructor
  SmallVector<const TypeBinding *> tmp(map.size(), nullptr);
  fillVectors(map, tmp, names, injected);
  params.reserve(map.size());
  for (auto *type : tmp) {
    assert(type);
    params.push_back(*type);
  }
}

ParamsStorage::ParamsStorage(const ParamsMap &map, size_t size, const TypeBinding &defaultBinding)
    : names(size), injected(size) {
  // Hack to get the bindings ordered without having a default constructor
  SmallVector<const TypeBinding *> tmp(size, nullptr);
  fillVectors(map, tmp, names, injected);
  params.reserve(size);
  for (auto *type : tmp) {
    if (type) {
      params.push_back(*type);
    } else {
      params.push_back(defaultBinding);
    }
  }
}

//==-----------------------------------------------------------------------==//
// Params
//==-----------------------------------------------------------------------==//

Params::operator ParamsMap() const {
  ParamsMap map;
  for (size_t i = 0; i < data()->params.size(); i++) {
    map.declare(data()->names[i], data()->params[i], i, data()->injected[i]);
  }
  return map;
}

const TypeBinding &Params::getParam(size_t i) const {
  assert(i < data()->params.size());
  return data()->params[i];
}

ArrayRef<ParamName> Params::getNames() const { return data()->names; }

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
    if (!data()->injected[i]) {
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

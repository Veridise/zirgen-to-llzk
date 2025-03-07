#include <cassert>
#include <iterator>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/Params.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

using namespace zhl;
using namespace mlir;

ParamsStorage::ParamsStorage() = default;

ParamsStorage::ParamsStorage(ParamsMap &map) : names(ParamNames(map.size())) {
  // Hack to get the bindings ordered without having a default constructor
  SmallVector<TypeBinding *> tmp(map.size());
  for (auto &entry : map) {
    auto pos = entry.getValue().second;
    tmp[pos] = &entry.getValue().first;
    names[pos] = entry.getKey();
  }
  for (auto *type : tmp) {
    params.push_back(*type);
  }
}

Params::Params(ParamsStorage &params) : sto(&params) {}
Params::Params(const ParamsStorage &params) : sto(&params) {}

MutableParams::MutableParams(ParamsStorage &params) : Params(params) {}

Params::operator ParamsMap() const {
  ParamsMap map;
  for (size_t i = 0; i < data()->params.size(); i++) {
    map.insert({data()->names[i], {data()->params[i], i}});
  }
  return map;
}

void Params::printNames(llvm::raw_ostream &os, char header, char footer) const {
  print<ParamName>(data()->names, os, [&](const auto &e) { os << e; }, header, footer);
}

void Params::printParams(llvm::raw_ostream &os, bool fullPrintout, char header, char footer) const {
  print<TypeBinding>(data()->params, os, [&](const auto &e) {
    e.print(os, fullPrintout);
  }, header, footer);
}

void Params::printMapping(llvm::raw_ostream &os, bool fullPrintout) const {
  os << "{ ";
  size_t c = 1;
  size_t siz = data()->params.size();
  for (size_t i = 0; i < siz; i++) {
    os << data()->names[i] << ": ";
    data()->params[i].print(os, fullPrintout);
    if (c < siz) {
      os << ", ";
    }
    c++;
  }
  os << " }";
}
template <typename Elt>
void Params::print(
    ArrayRef<Elt> lst, llvm::raw_ostream &os, std::function<void(const Elt &)> handler, char header,
    char footer
) const {
  if (data()->params.size() == 0) {
    return; // Don't print anything if there aren't any parameters
  }

  os << header;
  size_t c = 1;
  for (auto &e : lst) {
    handler(e);
    if (c < lst.size()) {
      os << ",";
    }
    c++;
  }
  os << footer;
}

TypeBinding Params::getParam(size_t i) const {
  assert(i < data()->params.size());
  return data()->params[i];
}

ArrayRef<ParamName> Params::getNames() const { return data()->names; }

MutableParams::iterator MutableParams::begin() const { return data()->params.begin(); }
Params::iterator Params::begin() const { return data()->params.begin(); }
MutableParams::iterator MutableParams::end() const { return data()->params.end(); }
Params::iterator Params::end() const { return data()->params.end(); }

bool Params::operator==(const Params &other) const {
  return data()->params == other.data()->params && data()->names == other.data()->names;
}

StringRef Params::getName(size_t i) const {
  assert(i < data()->names.size());
  return data()->names[i];
}

size_t Params::size() const { return data()->params.size(); }
mlir::MutableArrayRef<TypeBinding> MutableParams::getParams() const { return data()->params; }
mlir::ArrayRef<TypeBinding> Params::getParams() const { return data()->params; }

const TypeBinding *Params::operator[](StringRef name) const {
  for (size_t i = 0; i < data()->names.size(); i++) {
    if (data()->names[i] == name) {
      return &data()->params[i];
    }
  }
  return nullptr;
}

TypeBinding *MutableParams::operator[](StringRef name) const {
  auto ptr = Params::operator[](name);
  return const_cast<TypeBinding *>(ptr);
}

bool Params::empty() const { return data()->names.empty(); }

void MutableParams::replaceParam(StringRef name, const TypeBinding &binding) {
  auto found = this->operator[](name);
  if (found != nullptr) {
    found->print(llvm::dbgs() << "Replacing binding at address " << found << ":", true);
    llvm::dbgs() << "\n";
    *found = binding;
  }
}

bool Params::contains(StringRef name) const {
  auto B = data()->names.begin();
  auto E = data()->names.end();
  auto pos = std::find(B, E, name);
  return pos != E;
}

const ParamsStorage *Params::data() const { return sto; }

ParamsStorage *MutableParams::data() const { return const_cast<ParamsStorage *>(Params::data()); }

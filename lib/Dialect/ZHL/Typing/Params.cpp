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

ParamsStorage::ParamsStorage(ParamsMap map) : names(ParamNames(map.size())) {
  // Hack to get the bindings ordered without having a default constructor
  std::vector<TypeBinding *> tmp(map.size());
  for (auto &[k, v] : map) {
    tmp[v.second] = &v.first;
    names[v.second] = k;
  }
  for (auto *type : tmp) {
    params.push_back(*type);
  }
}

Params::Params(ParamsStorage &params) : sto(&params) {}

Params::operator ParamsMap() const {
  ParamsMap map;
  for (size_t i = 0; i < sto->params.size(); i++) {
    map.insert({sto->names[i], {sto->params[i], i}});
  }
  return map;
}

void Params::printNames(llvm::raw_ostream &os, char header, char footer) const {
  print<ParamName>(sto->names, os, [&](const auto &e) { os << e; }, header, footer);
}

void Params::printParams(llvm::raw_ostream &os, bool fullPrintout, char header, char footer) const {
  print<TypeBinding>(sto->params, os, [&](const auto &e) {
    e.print(os, fullPrintout);
  }, header, footer);
}

void Params::printMapping(llvm::raw_ostream &os, bool fullPrintout) const {
  os << "{ ";
  size_t c = 1;
  size_t siz = sto->params.size();
  for (size_t i = 0; i < siz; i++) {
    os << sto->names[i] << ": ";
    sto->params[i].print(os, fullPrintout);
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
  if (sto->params.size() == 0) {
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
  assert(i < sto->params.size());
  return sto->params[i];
}

ArrayRef<ParamName> Params::getNames() const { return sto->names; }

Params::iterator Params::begin() { return sto->params.begin(); }
Params::const_iterator Params::begin() const { return sto->params.begin(); }
Params::iterator Params::end() { return sto->params.end(); }
Params::const_iterator Params::end() const { return sto->params.end(); }

bool Params::operator==(const Params &other) const {
  return sto->params == other.sto->params && sto->names == other.sto->names;
}

StringRef Params::getName(size_t i) const {
  assert(i < sto->names.size());
  return sto->names[i];
}

size_t Params::size() const { return sto->params.size(); }
mlir::MutableArrayRef<TypeBinding> Params::getParams() { return sto->params; }
mlir::ArrayRef<TypeBinding> Params::getParams() const { return sto->params; }

const TypeBinding *Params::operator[](StringRef name) const {
  for (size_t i = 0; i < sto->names.size(); i++) {
    if (sto->names[i] == name) {
      return &sto->params[i];
    }
  }
  return nullptr;
}

TypeBinding *Params::operator[](StringRef name) {
  for (size_t i = 0; i < sto->names.size(); i++) {
    if (sto->names[i] == name) {
      return &sto->params[i];
    }
  }
  return nullptr;
}
bool Params::empty() const { return sto->names.empty(); }

void Params::replaceParam(StringRef name, const TypeBinding &binding) {
  auto found = this->operator[](name);
  if (found != nullptr) {
    *found = binding;
  }
}

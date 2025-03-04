#include <cassert>
#include <iterator>
#include <mlir/IR/Types.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/Frame.h>
#include <zklang/Dialect/ZHL/Typing/FrameSlot.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

using namespace zhl;
using namespace mlir;

Params::Params(ParamsMap map) : names(ParamNames(map.size())) {
  // Hack to get the bindings ordered without having a default constructor
  std::vector<TypeBinding *> tmp(map.size());
  for (auto &[k, type] : map) {
    tmp[k.second] = &type;
    names[k.second] = k.first;
  }
  for (auto *type : tmp) {
    params.push_back(*type);
  }
}

Params::Params() = default;

Params::operator ParamsMap() const {
  ParamsMap map;
  for (size_t i = 0; i < params.size(); i++) {
    map.insert({{names[i], i}, params.at(i)});
  }
  return map;
}

void Params::printNames(llvm::raw_ostream &os, char header, char footer) const {
  print<std::string>(names, os, [&](const auto &e) { os << e; }, header, footer);
}

void Params::printParams(llvm::raw_ostream &os, bool fullPrintout, char header, char footer) const {
  print<TypeBinding>(params, os, [&](const auto &e) { e.print(os, fullPrintout); }, header, footer);
}

void Params::printMapping(llvm::raw_ostream &os, bool fullPrintout) const {
  os << "{ ";
  size_t c = 1;
  size_t siz = params.size();
  for (size_t i = 0; i < siz; i++) {
    os << names[i] << ": ";
    params.at(i).print(os, fullPrintout);
    if (c < siz) {
      os << ", ";
    }
    c++;
  }
  os << " }";
}
template <typename Elt>
void Params::print(
    const std::vector<Elt> &lst, llvm::raw_ostream &os, std::function<void(const Elt &)> handler,
    char header, char footer
) const {
  if (params.size() == 0) {
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
  assert(i < params.size());
  return params[i];
}

ArrayRef<std::string> Params::getNames() const { return names; }

Params::ParamsList::iterator Params::begin() { return params.begin(); }
Params::ParamsList::const_iterator Params::begin() const { return params.begin(); }
Params::ParamsList::iterator Params::end() { return params.end(); }
Params::ParamsList::const_iterator Params::end() const { return params.end(); }

bool Params::operator==(const Params &other) const {
  return params == other.params && names == other.names;
}

std::string_view zhl::Params::getName(size_t i) const {
  assert(i < names.size());
  return names[i];
}

size_t zhl::Params::size() const { return params.size(); }
mlir::ArrayRef<TypeBinding> zhl::Params::getParams() const { return params; }

const TypeBinding *zhl::Params::operator[](std::string_view name) const {
  for (size_t i = 0; i < names.size(); i++) {
    if (names.at(i) == name) {
      return &params.at(i);
    }
  }
  return nullptr;
}

TypeBinding *zhl::Params::operator[](std::string_view name) {
  for (size_t i = 0; i < names.size(); i++) {
    if (names.at(i) == name) {
      return &params.at(i);
    }
  }
  return nullptr;
}
bool zhl::Params::empty() const { return names.empty(); }
void zhl::Params::replaceParam(std::string_view name, const TypeBinding &binding) {
  auto found = this->operator[](name);
  if (found != nullptr) {
    *found = binding;
  }
}

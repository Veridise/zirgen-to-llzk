#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>

namespace llvm {
class raw_ostream;
}

namespace zhl {

class Params {
public:
  using iterator = mlir::SmallVector<TypeBinding, 0>::iterator;
  using const_iterator = mlir::SmallVector<TypeBinding>::const_iterator;

  Params(ParamsStorage &);
  size_t size() const;

  operator ParamsMap() const;

  mlir::StringRef getName(size_t i) const;

  TypeBinding getParam(size_t i) const;

  mlir::ArrayRef<ParamName> getNames() const;
  mlir::MutableArrayRef<TypeBinding> getParams();
  mlir::ArrayRef<TypeBinding> getParams() const;

  const TypeBinding *operator[](mlir::StringRef name) const;
  TypeBinding *operator[](mlir::StringRef name);

  void printMapping(llvm::raw_ostream &os, bool fullPrintout = false) const;
  void printNames(llvm::raw_ostream &os, char header = '<', char footer = '>') const;
  void printParams(
      llvm::raw_ostream &os, bool fullPrintout = false, char header = '<', char footer = '>'
  ) const;

  iterator begin();
  const_iterator begin() const;
  iterator end();
  const_iterator end() const;

  bool empty() const;

  void replaceParam(mlir::StringRef name, const TypeBinding &binding);

  bool operator==(const Params &) const;

private:
  template <typename Elt>
  void print(
      mlir::ArrayRef<Elt> lst, llvm::raw_ostream &os, std::function<void(const Elt &)> handler,
      char header, char footer
  ) const;

  ParamsStorage *sto;
};

} // namespace zhl

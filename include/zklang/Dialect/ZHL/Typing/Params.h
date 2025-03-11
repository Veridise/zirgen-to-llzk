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
  using iterator = mlir::SmallVector<TypeBinding>::const_iterator;

  Params(const ParamsStorage &);
  Params(ParamsStorage &);
  size_t size() const;
  size_t sizeOfDeclared() const;

  operator ParamsMap() const;

  mlir::StringRef getName(size_t i) const;

  const TypeBinding &getParam(size_t i) const;

  mlir::ArrayRef<ParamName> getNames() const;
  mlir::ArrayRef<TypeBinding> getParams() const;
  mlir::SmallVector<TypeBinding, 0> getDeclaredParams() const;

  const TypeBinding *operator[](mlir::StringRef name) const;

  void printMapping(llvm::raw_ostream &os, bool fullPrintout = false) const;
  void printNames(llvm::raw_ostream &os, char header = '<', char footer = '>') const;
  void printParams(
      llvm::raw_ostream &os, bool fullPrintout = false, char header = '<', char footer = '>'
  ) const;

  iterator begin() const;
  iterator end() const;

  bool contains(mlir::StringRef) const;

  bool empty() const;

  bool operator==(const Params &) const;

  const ParamsStorage *data() const;

protected:
private:
  template <typename Elt>
  void print(
      mlir::ArrayRef<Elt> lst, llvm::raw_ostream &os, std::function<void(const Elt &)> handler,
      char header, char footer
  ) const;

  const ParamsStorage *sto;
};

class MutableParams : public Params {
public:
  using iterator = mlir::SmallVector<TypeBinding, 0>::iterator;
  MutableParams(ParamsStorage &);

  void replaceParam(mlir::StringRef name, const TypeBinding &binding);

  TypeBinding &getParam(size_t i) const;
  mlir::MutableArrayRef<TypeBinding> getParams() const;
  TypeBinding *operator[](mlir::StringRef name) const;

  iterator begin() const;
  iterator end() const;
  ParamsStorage *data() const;
};

} // namespace zhl

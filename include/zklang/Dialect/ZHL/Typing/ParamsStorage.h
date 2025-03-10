#pragma once

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>

namespace zhl {

struct ParamData {
  bool Injected;
  TypeBinding Type;
  uint64_t Pos;

  ParamData(const TypeBinding &T, uint64_t P, bool I) : Injected(I), Type(T), Pos(P) {}
  ParamData(const TypeBinding &T, uint64_t P) : ParamData(T, P, false) {}
};

using ParamName = std::string;
using ParamsList = mlir::SmallVector<TypeBinding, 0>;
using ParamNames = mlir::SmallVector<ParamName>;

/// Wrapper around a mapping between parameter names and their information.
/// Its goal is to make the StringMap easier to use and not require too many initializer lists.
class ParamsMap {
public:
  using iterator = llvm::StringMap<ParamData>::iterator;
  using const_iterator = llvm::StringMap<ParamData>::const_iterator;

  ParamsMap &declare(mlir::StringRef name, const TypeBinding &type, uint64_t pos, bool injected) {
    impl.insert({name, ParamData(type, pos, injected)});
    return *this;
  }

  ParamsMap &declare(mlir::StringRef name, const TypeBinding &type, uint64_t pos) {
    return declare(name, type, pos, false);
  }

  ParamsMap &declare(mlir::StringRef name, const TypeBinding &type) {
    return declare(name, type, size(), false);
  }

  size_t size() const { return impl.size(); }

  iterator begin() { return impl.begin(); }
  const_iterator begin() const { return impl.begin(); }
  iterator end() { return impl.end(); }
  const_iterator end() const { return impl.end(); }

  bool contains(mlir::StringRef k) const { return impl.contains(k); }

private:
  llvm::StringMap<ParamData> impl;
};

struct ParamsStorage {
  ParamsList params;
  ParamNames names;
  llvm::BitVector injected;

  ParamsStorage();
  ParamsStorage(ParamsMap &map);

  bool operator==(const ParamsStorage &) const;
};

} // namespace zhl

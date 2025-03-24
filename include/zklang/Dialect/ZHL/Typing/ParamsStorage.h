#pragma once

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>

namespace zhl {

struct ParamData {
  /// Marks the parameter as injected, which means that it was not declared by the input ZIR code
  bool Injected;
  /// The type associated with the parameter
  TypeBinding Type;
  /// The position in the parameters list where this parameter was declared
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

  /// Adds a parameter to the map.
  ParamsMap &declare(mlir::StringRef name, const TypeBinding &type, uint64_t pos, bool injected) {
    impl.insert({name, ParamData(type, pos, injected)});
    return *this;
  }

  /// Adds a parameter to the map that is automatically marked as non-injected
  ParamsMap &declare(mlir::StringRef name, const TypeBinding &type, uint64_t pos) {
    return declare(name, type, pos, false);
  }

  /// Adds a parameter to the map selecting automatically its position and marking it as
  /// non-injected
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

/// Holds the information related to the parameters of a type binding as a set of ordered lists.
/// The goal of this abstraction is to facilitate iteration and random access over the elements
struct ParamsStorage {
  ParamsList params;
  ParamNames names;
  /// Used to denote if the n-th parameter has been injected or it was declared.
  llvm::BitVector injected;

  /// Default constructs an empty list of parameters.
  ParamsStorage() = default;
  /// Constructs the list of parameters from the given map. Expects it to be densely filled, meaning
  /// that from 0 to the size of the map minus 1 there is a parameter assigned to that position.
  ParamsStorage(const ParamsMap &Map);
  /// Constructs the list of parameters from the given map but sets the total size to the given size
  /// and fills the positions that are not claimed by the parameters in the map with a default
  /// value.
  ParamsStorage(const ParamsMap &Map, size_t Size, const TypeBinding &Default);
};

} // namespace zhl

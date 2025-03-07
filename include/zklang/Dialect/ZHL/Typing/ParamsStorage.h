#pragma once

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>

namespace zhl {

using ParamName = std::string;
using ParamsList = mlir::SmallVector<TypeBinding, 0>;
using ParamNames = mlir::SmallVector<ParamName>;
using ParamsMap = llvm::StringMap<std::pair<TypeBinding, uint64_t>>;

struct ParamsStorage {
  ParamsList params;
  ParamNames names;

  ParamsStorage();
  ParamsStorage(ParamsMap &map);

  bool operator==(const ParamsStorage &) const;
};

} // namespace zhl

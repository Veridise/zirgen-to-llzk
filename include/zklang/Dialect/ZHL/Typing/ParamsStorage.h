#pragma once

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Support/LLVM.h>
#include <zklang/Dialect/ZHL/Typing/TypeBinding.h>

namespace zhl {

using ParamName = mlir::SmallString<10>;
using ParamsList = mlir::SmallVector<TypeBinding, 0>;
using ParamNames = mlir::SmallVector<ParamName>;
using ParamsMap = llvm::StringMap<std::pair<TypeBinding, uint64_t>>;

struct ParamsStorage {
  ParamsList params;
  ParamNames names;

  ParamsStorage(ParamsMap map);
};

} // namespace zhl

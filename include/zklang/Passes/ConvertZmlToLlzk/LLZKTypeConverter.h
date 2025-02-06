#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include <string_view>
#include <unordered_set>

namespace llzk {

class LLZKTypeConverter : public mlir::TypeConverter {
public:
  LLZKTypeConverter(mlir::Operation *);

private:
  std::unordered_set<std::string_view> feltEquivalentTypes;
  mlir::Operation *symbolTableRoot;
  mlir::SymbolTableCollection stc;
};

} // namespace llzk

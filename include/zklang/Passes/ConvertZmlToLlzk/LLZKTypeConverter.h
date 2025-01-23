#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include <string_view>
#include <unordered_set>

namespace llzk {

class LLZKTypeConverter : public mlir::TypeConverter {
public:
  LLZKTypeConverter(std::unordered_set<std::string_view> builtinOverrideSet);

private:
  std::unordered_set<std::string_view> feltEquivalentTypes;
};

} // namespace llzk

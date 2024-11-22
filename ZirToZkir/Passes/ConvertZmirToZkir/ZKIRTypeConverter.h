#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace zkir {

class ZKIRTypeConverter : public mlir::TypeConverter {
public:
  ZKIRTypeConverter();

private:
};

} // namespace zkir

#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace zkc::Zmir {

class ZMIRTypeConverter : public mlir::TypeConverter {
public:
  ZMIRTypeConverter();

private:
};

} // namespace zkc::Zmir

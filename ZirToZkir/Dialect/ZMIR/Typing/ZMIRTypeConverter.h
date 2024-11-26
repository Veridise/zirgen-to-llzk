#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace zkc::Zmir {

class ZMIRTypeConverter : public mlir::TypeConverter {
public:
  ZMIRTypeConverter();

private:
  mlir::FailureOr<mlir::Value> findTypeInUseDefChain(mlir::Value, mlir::Type);
};

} // namespace zkc::Zmir

#pragma once

#include "ZirToZkir/Dialect/ZHL/Typing/TypeBindings.h"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

namespace zkc::Zmir {

mlir::Type materializeTypeBinding(mlir::MLIRContext *, const zhl::TypeBinding &);
mlir::FunctionType materializeTypeBindingConstructor(mlir::OpBuilder &, const zhl::TypeBinding &);

} // namespace zkc::Zmir

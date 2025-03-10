#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <zklang/Dialect/ZHL/Typing/TypeBindings.h>

namespace zml {

mlir::Type materializeTypeBinding(mlir::MLIRContext *, const zhl::TypeBinding &);
mlir::FunctionType materializeTypeBindingConstructor(
    mlir::OpBuilder &, const zhl::TypeBinding &, bool callerPov = false
);

} // namespace zml

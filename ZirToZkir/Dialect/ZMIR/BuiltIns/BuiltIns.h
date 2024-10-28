#pragma once

#include <mlir/IR/Builders.h>

namespace zkc::Zmir {

// Add builtin components using the given builder
void addBuiltins(mlir::OpBuilder &builder);

} // namespace zkc::Zmir

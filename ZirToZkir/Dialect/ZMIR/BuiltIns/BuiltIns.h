#pragma once

#include <mlir/IR/Builders.h>

namespace zkc::Zmir {

// Add builtin components using the given builder
void addBuiltins(mlir::OpBuilder &, std::function<bool(mlir::StringRef)>);

} // namespace zkc::Zmir

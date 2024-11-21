#pragma once

#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include <cstdint>
#include <vector>

namespace zkc {

struct ComponentArity {
  /// The last argument of a ZIR component can be variadic.
  bool isVariadic;
  uint32_t paramCount;
  std::vector<mlir::Location> locs;

  ComponentArity();
};

/// Searches all Zhl::ConstructorParamOp in the component
/// and returns the largest index declared by the ops.
ComponentArity getComponentConstructorArity(zirgen::Zhl::ComponentOp);

} // namespace zkc

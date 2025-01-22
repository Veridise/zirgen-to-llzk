#pragma once

#include <mlir/Support/LogicalResult.h>
namespace zklang {

/// ZIR frontend driver
mlir::LogicalResult zirDriver(int &, char **&);

} // namespace zklang

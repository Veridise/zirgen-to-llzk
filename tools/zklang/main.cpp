#include "zklang/Frontends/ZIR/Driver.h"
#include <mlir/Support/LogicalResult.h>

int main(int argc, char **argv) { return mlir::failed(zklang::zirDriver(argc, argv)); }

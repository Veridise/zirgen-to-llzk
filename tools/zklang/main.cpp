#include <mlir/Support/LogicalResult.h>
#include <zklang/Frontends/ZIR/Driver.h>

int main(int argc, char **argv) { return mlir::failed(zklang::zirDriver(argc, argv)); }

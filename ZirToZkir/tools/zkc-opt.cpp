#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<zkc::Zmir::ZmirDialect>();
  registry.insert<zirgen::Zhl::ZhlDialect>();
  return failed(mlir::MlirOptMain(
      argc, argv, "ZIR to ZKIR transformation pipeline\n", registry));
}

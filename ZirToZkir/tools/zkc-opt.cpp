#include "ZirToZkir/Dialect/ZMIR/IR/Dialect.h"
#include "ZirToZkir/Dialect/ZMIR/Transforms/Passes.h"
#include "ZirToZkir/Passes/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zkir/Dialect/ZKIR/IR/Dialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  mlir::registerCSEPass();
  mlir::registerCanonicalizer();

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createReconcileUnrealizedCastsPass();
  });
  zkc::registerPasses();
  zkc::Zmir::registerPasses();

  registry.insert<zkc::Zmir::ZmirDialect>();
  registry.insert<zirgen::Zhl::ZhlDialect>();
  registry.insert<zkir::ZKIRDialect>();
  return failed(mlir::MlirOptMain(argc, argv, "ZIR to ZKIR transformation pipeline\n", registry));
}

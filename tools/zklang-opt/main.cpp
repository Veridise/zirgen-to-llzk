#include <llzk/Dialect/LLZK/IR/Dialect.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zklang/Dialect/ZHL/Typing/Passes.h>
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/Transforms/Passes.h>
#include <zklang/Passes/Passes.h>

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  mlir::registerCSEPass();
  mlir::registerCanonicalizer();

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createReconcileUnrealizedCastsPass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createTopologicalSortPass();
  });
  zklang::registerPasses();
  zml::registerPasses();
  zhl::registerPasses();

  registry.insert<zml::ZMLDialect>();
  registry.insert<zirgen::Zhl::ZhlDialect>();
  registry.insert<llzk::LLZKDialect>();
  return failed(mlir::MlirOptMain(argc, argv, "zklang optimizer\n", registry));
}

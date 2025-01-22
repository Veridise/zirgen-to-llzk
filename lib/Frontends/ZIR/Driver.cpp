#include "zklang/Frontends/ZIR/Driver.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "zklang/Dialect/ZML/BuiltIns/BuiltIns.h"

#include "zirgen/dsl/lower.h"
#include "zirgen/dsl/parser.h"
/*#include "zirgen/dsl/passes/Passes.h"*/
/*#include "zirgen/dsl/stats.h"*/
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Transforms/Passes.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zklang/Dialect/ZHL/Typing/Passes.h"
#include "zklang/Dialect/ZML/IR/Dialect.h"
#include "zklang/Dialect/ZML/Transforms/Passes.h"
#include "zklang/Passes/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <mlir/Support/LogicalResult.h>

namespace cl = llvm::cl;
using namespace mlir;

static cl::opt<std::string> inputFilename(
    cl::Positional, cl::desc("<input zirgen file>"), cl::value_desc("filename"), cl::Required
);

namespace {
enum class Action {
  None = 0,
  PrintAST,
  PrintZHL,
  PrintZML,
  OptimizeZML,
  PrintLlzk,
};
} // namespace

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("The kind of output desired"),
    cl::values(
        clEnumValN(Action::PrintAST, "ast", "Output the AST"),
        clEnumValN(Action::PrintZHL, "zhl", "Output untyped high level ZIR IR"),
        clEnumValN(Action::PrintZML, "zml", "Output typed medium level ZIR IR"),
        clEnumValN(
            Action::OptimizeZML, "zmlopt",
            "Output typed medium level ZIR IR with separate compute and constrain functions"
        ),
        clEnumValN(Action::PrintLlzk, "llzk", "Output LLZK IR")
    )
);

static cl::list<std::string> includeDirs("I", cl::desc("Add include path"), cl::value_desc("path"));

class Driver {
public:
  static FailureOr<std::unique_ptr<Driver>> Make(int &argc, char **argv);
  LogicalResult run();

private:
  Driver(int &argc, char **argv);
  void openMainFile(std::string);
  zirgen::dsl::ast::Module::Ptr parse();
  std::optional<mlir::ModuleOp> lowerToZhl(zirgen::dsl::ast::Module &);
  void configureLoweringPipeline();

  llvm::InitLLVM llvm;
  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
  llvm::SourceMgr sourceManager;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler;
  zirgen::dsl::Parser parser;
  mlir::PassManager pm;
};

FailureOr<std::unique_ptr<Driver>> Driver::Make(int &argc, char **argv) {
  // Done this way to keep the constructor private
  auto driver = std::unique_ptr<Driver>(new Driver(argc, argv));

  applyDefaultTimingPassManagerCLOptions(driver->pm);
  if (failed(applyPassManagerCLOptions(driver->pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n";
    return failure();
  }
  driver->pm.enableVerifier(true);
  return std::move(driver);
}

Driver::Driver(int &argc, char **argv)
    : llvm(argc, argv), registry{}, context(registry), sourceManager{},
      sourceMgrHandler(sourceManager, &context), parser(sourceManager), pm(&context) {
  sourceManager.setIncludeDirs(includeDirs);

  parser.addPreamble(zkc::Zmir::zirPreamble);
}

void Driver::configureLoweringPipeline() {
  pm.clear();
  auto modPipeline = pm.nest<mlir::ModuleOp>();
  modPipeline.addPass(zkc::createStripTestsPass());
  modPipeline.addPass(zkc::createStripDirectivesPass());
  modPipeline.addPass(zkc::Zmir::createInjectBuiltInsPass());
  modPipeline.addPass(zkc::createConvertZhlToZmirPass());
  modPipeline.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (emitAction == Action::PrintZML) {
    return;
  }

  modPipeline.nest<zkc::Zmir::ComponentOp>().nest<mlir::func::FuncOp>().addPass(
      zkc::Zmir::createLowerBuiltInsPass()
  );
  modPipeline.addPass(zkc::Zmir::createRemoveBuiltInsPass());
  modPipeline.addPass(zkc::Zmir::createSplitComponentBodyPass());
  auto splitCompFuncsPipeline =
      modPipeline.nest<zkc::Zmir::SplitComponentOp>().nest<mlir::func::FuncOp>();
  splitCompFuncsPipeline.addPass(zkc::Zmir::createRemoveIllegalComputeOpsPass());
  splitCompFuncsPipeline.addPass(zkc::Zmir::createRemoveIllegalConstrainOpsPass());
  splitCompFuncsPipeline.addPass(mlir::createCSEPass());

  if (emitAction == Action::OptimizeZML) {
    return;
  }

  modPipeline.addPass(zkc::createConvertZmirComponentsToLlzkPass());
  auto llzkStructPipeline = modPipeline.nest<llzk::StructDefOp>();
  llzkStructPipeline.addPass(zkc::createConvertZmirToLlzkPass());
  llzkStructPipeline.addPass(mlir::createReconcileUnrealizedCastsPass());
  llzkStructPipeline.addPass(mlir::createCanonicalizerPass());
}

zirgen::dsl::ast::Module::Ptr Driver::parse() {
  auto ast = parser.parseModule();
  if (!ast) {
    const auto &errors = parser.getErrors();
    for (const auto &error : errors) {
      sourceManager.PrintMessage(llvm::errs(), error);
    }
    llvm::errs() << "parsing failed with " << errors.size() << " errors\n";
    return nullptr;
  }
  return ast;
}
std::optional<mlir::ModuleOp> Driver::lowerToZhl(zirgen::dsl::ast::Module &mod) {
  return zirgen::dsl::lower(context, sourceManager, &mod);
}

LogicalResult Driver::run() {
  openMainFile(inputFilename);
  auto ast = parse();
  if (!ast) {
    return failure();
  }
  if (emitAction == Action::PrintAST) {
    ast->print(llvm::outs());
    return success();
  }

  auto mod = lowerToZhl(*ast);
  if (!mod) {
    return failure();
  }
  if (emitAction == Action::PrintZHL) {
    mod->print(llvm::outs());
    return success();
  }

  configureLoweringPipeline();
  if (failed(pm.run(*mod))) {
    llvm::errs() << "an internal compiler error ocurred while lowering this module:\n";
    mod->print(llvm::errs());
    return failure();
  }

  mod->print(llvm::outs());

  return success();
}

void Driver::openMainFile(std::string filename) {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code error = fileOrErr.getError()) {
    sourceManager.PrintMessage(
        llvm::SMLoc(), llvm::SourceMgr::DiagKind::DK_Error, "could not open input file " + filename
    );
  }
  sourceManager.AddNewSourceBuffer(std::move(*fileOrErr), mlir::SMLoc());
}

namespace zklang {

LogicalResult zirDriver(int &argc, char **argv) {
  auto driver = Driver::Make(argc, argv);
  if (failed(driver)) {
    return failure();
  }
  return (*driver)->run();
}

} // namespace zklang

#include "zklang/Frontends/ZIR/Driver.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "zklang/Dialect/ZML/BuiltIns/BuiltIns.h"
#include "llvm/Support/WithColor.h"

#include "zirgen/dsl/lower.h"
#include "zirgen/dsl/parser.h"
/*#include "zirgen/dsl/passes/Passes.h"*/
/*#include "zirgen/dsl/stats.h"*/
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Debug/CLOptionsSetup.h"
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
#include "llvm/Support/raw_ostream.h"
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "zir-driver"

namespace cl = llvm::cl;
using namespace mlir;

static cl::opt<std::string> inputFilename(
    cl::Positional, cl::desc("<input zir file>"), cl::value_desc("filename"), cl::Required
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
static cl::opt<std::string>
    outputFile("o", cl::desc("Where to write the result"), cl::value_desc("output"));

// Dev-oriented command line options only available in debug builds
// They are defined with external storage to avoid having to wrap
// the code that uses them in preprocessor checks too
bool DisableMultiThreadingFlag = false;

#ifndef NDEBUG
static cl::opt<bool, true> DisableMultiThreading(
    "disable-multithreading", cl::desc("Disable multithreading of the lowering pipeline"),
    cl::Hidden, cl::location(DisableMultiThreadingFlag)
);
#endif

/// A wrapper around the dialect registry that ensures that the required dialects are available
/// at initialization.
struct ZirFrontendDialects {

  ZirFrontendDialects() {
    registry.insert<zkc::Zmir::ZmirDialect>();
    registry.insert<zirgen::Zhl::ZhlDialect>();
    registry.insert<llzk::LLZKDialect>();
  }
  mlir::DialectRegistry registry;
};

class Driver {
public:
  static FailureOr<std::unique_ptr<Driver>> Make(int &argc, char **&argv);
  LogicalResult run();

private:
  Driver(int &argc, char **&argv);
  void openMainFile(std::string);
  zirgen::dsl::ast::Module::Ptr parse();
  std::optional<mlir::ModuleOp> lowerToZhl(zirgen::dsl::ast::Module &);
  void configureLoweringPipeline();

  llvm::InitLLVM llvm;
  ZirFrontendDialects dialects;
  mlir::MLIRContext context;
  llvm::SourceMgr sourceManager;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler;
  mlir::PassManager pm;
};

FailureOr<std::unique_ptr<Driver>> Driver::Make(int &argc, char **&argv) {
  // Done this way to keep the constructor private
  std::unique_ptr<Driver> driver(new Driver(argc, argv));

  applyDefaultTimingPassManagerCLOptions(driver->pm);
  if (failed(applyPassManagerCLOptions(driver->pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n";
    return failure();
  }
  driver->pm.enableVerifier(true);
  return std::move(driver);
}

Driver::Driver(int &argc, char **&argv)
    : llvm(argc, argv), dialects(), context(dialects.registry), sourceManager{},
      sourceMgrHandler(sourceManager, &context), pm(&context) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::tracing::DebugConfig::registerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "zklang ZIR frontend\n");

  context.loadAllAvailableDialects();

  if (DisableMultiThreadingFlag) {
    llvm::errs() << "Multithreading was disabled!\n";
    context.disableMultithreading();
  }
}

void Driver::configureLoweringPipeline() {
  pm.clear();
  if (emitAction >= Action::PrintLlzk) {
    pm.addPass(zkc::createInjectLlzkModAttrsPass());
  }
  pm.addPass(zkc::createStripTestsPass());
  pm.addPass(zkc::Zmir::createInjectBuiltInsPass());
  pm.addPass(zkc::createConvertZhlToZmirPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (emitAction == Action::PrintZML) {
    return;
  }

  pm.nest<zkc::Zmir::ComponentOp>().nest<mlir::func::FuncOp>().addPass(
      zkc::Zmir::createLowerBuiltInsPass()
  );
  pm.addPass(zkc::Zmir::createRemoveBuiltInsPass());
  pm.addPass(zkc::Zmir::createSplitComponentBodyPass());
  auto &splitCompPipeline = pm.nest<zkc::Zmir::SplitComponentOp>();
  auto &splitCompFuncsPipeline = splitCompPipeline.nest<mlir::func::FuncOp>();
  splitCompFuncsPipeline.addPass(zkc::Zmir::createRemoveIllegalComputeOpsPass());
  splitCompFuncsPipeline.addPass(zkc::Zmir::createRemoveIllegalConstrainOpsPass());
  splitCompFuncsPipeline.addPass(mlir::createCSEPass());

  if (emitAction == Action::OptimizeZML) {
    return;
  }

  pm.addPass(zkc::createConvertZmlToLlzkPass());
  auto &llzkStructPipeline = pm.nest<llzk::StructDefOp>();
  llzkStructPipeline.addPass(mlir::createReconcileUnrealizedCastsPass());
  llzkStructPipeline.addPass(mlir::createCanonicalizerPass());
}

zirgen::dsl::ast::Module::Ptr Driver::parse() {
  zirgen::dsl::Parser parser(sourceManager);
  parser.addPreamble(zkc::Zmir::zirPreamble);
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

/// Ensures the module is erased no matter the lowerig result.
class ModuleEraseGuard {
public:
  ModuleEraseGuard(mlir::ModuleOp &op) : mod(op) {}
  ~ModuleEraseGuard() { mod.erase(); }

private:
  mlir::ModuleOp mod;
};

// Simple RAII for handling the destination stream
class Dst {
public:
  Dst() : dst(&llvm::outs()) {
    std::string out = outputFile;
    if (out.empty()) {
      std::string base = inputFilename;
      out = base + ".mlir";
    }
    if (out != "-") {
      llvm::dbgs() << "Writing result to " << out << "\n";
      dst = new llvm::raw_fd_ostream(out, EC);
    }
  }
  ~Dst() {
    if (dst != &llvm::outs()) {
      delete dst;
    }
  }
  const std::error_code &error() { return EC; }

  llvm::raw_ostream &operator*() { return *dst; }

private:
  std::error_code EC;
  llvm::raw_ostream *dst;
};

LogicalResult Driver::run() {
  Dst dst;
  if (dst.error()) {
    llvm::errs() << "Failed to open output file: " << dst.error().message() << "\n";
    return failure();
  }
  openMainFile(inputFilename);
  sourceManager.setIncludeDirs(includeDirs);

  auto ast = parse();
  if (!ast) {
    return failure();
  }
  if (emitAction == Action::PrintAST) {
    ast->print(*dst);
    return success();
  }

  auto mod = lowerToZhl(*ast);
  if (!mod) {
    return failure();
  }
  ModuleEraseGuard guard(*mod);
  if (emitAction == Action::PrintZHL) {
    mod->print(*dst);
    return success();
  }

  configureLoweringPipeline();
  pm.dump();
  if (failed(pm.run(*mod))) {
    DEBUG_WITH_TYPE("zir-driver-dump-on-error", llvm::errs() << "Module contents:\n";
                    mod->print(llvm::errs()));
    llvm::WithColor(llvm::errs(), llvm::raw_ostream::RED, true)
        << "An internal compiler error ocurred while lowering this module.\n";
    return failure();
  }

  mod->print(*dst);

  llvm::WithColor(llvm::errs(), llvm::raw_ostream::GREEN) << "Success!\n";
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

LogicalResult zirDriver(int &argc, char **&argv) {

  FailureOr<std::unique_ptr<Driver>> driver = Driver::Make(argc, argv);
  if (failed(driver)) {
    return failure();
  }
  return (*driver)->run();
}

} // namespace zklang

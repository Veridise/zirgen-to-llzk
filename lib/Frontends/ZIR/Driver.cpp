#include "zklang/Frontends/ZIR/Driver.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "zirgen/dsl/lower.h"
#include "zirgen/dsl/parser.h"
#include "zirgen/dsl/passes/Passes.h"
#include "zirgen/dsl/stats.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

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
  Driver(int argc, char **argv);

  LogicalResult run();

private:
  void openMainFile(std::string);
  mlir::FailureOr<zirgen::dsl::ast::Module::Ptr> parse();
  std::optional<mlir::ModuleOp> lowerToZhl(zirgen::dsl::ast::Module &);

  llvm::InitLLVM llvm;
  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
  llvm::SourceMgr sourceManager;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler;
  zirgen::dsl::Parser parser;
};

class StepBase {
public:
  virtual LogicalResult run() = 0;
};

template <typename In, typename Out> class Step : public StepBase {
public:
  template <typename Out2> Step<Out, Out2> then(std::function<Out2(Out, Driver &)> nextAction) {}

private:
  Action goal;
  std::function<Out(In, Driver &)> action
};

Driver::Driver(int argc, char **argv)
    : llvm(argv, argv), registry{}, context(registry), sourceManager{},
      sourceMgrHandler(sourceManager, &context), parser(sourceManager) {
  sourceManager.setIncludeDirs(includeDirs);

  parser.addPreamble(zirgen::Typing::getBuiltinPreamble());
}

void dumpResultObj(mlir::FailureOr<zirgen::dsl::ast::Module::Ptr> &ast) {
  ast->print(llvm::outs());
}

LogicalResult Driver::run() {
  openMainFile(inputFilename);
  auto ast = parse();
  if (failed(ast)) {
    return failure();
  }
  if (emitAction == Action::PrintAST) {
    ast->print(llvm::outs());
    return success();
  }

  auto zhlModule = lowerToZhl(**ast);
  if (!zhlModule) {
    return failure();
  }
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

LogicalResult zirDriver(int argc, char **argv) {
  Driver driver(argc, argv);
  return driver.run();
}

} // namespace zklang

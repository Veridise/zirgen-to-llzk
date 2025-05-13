//===- Driver.cpp - Zirgen driver ------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Support/raw_ostream.h>
#include <llzk/Dialect/InitDialects.h>
#include <llzk/Dialect/Struct/IR/Dialect.h>
#include <mlir/Bytecode/BytecodeWriter.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Debug/CLOptionsSetup.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>
#include <optional>
#include <zirgen/Dialect/ZHL/IR/ZHL.h>
#include <zirgen/dsl/lower.h>
#include <zirgen/dsl/parser.h>
#include <zklang/Dialect/ZHL/Typing/ParamsStorage.h>
#include <zklang/Dialect/ZHL/Typing/Passes.h>
#include <zklang/Dialect/ZML/BuiltIns/BuiltIns.h>
#include <zklang/Dialect/ZML/IR/Dialect.h>
#include <zklang/Dialect/ZML/Transforms/Passes.h>
#include <zklang/Frontends/ZIR/Driver.h>
#include <zklang/Passes/Passes.h>

#define DEBUG_TYPE "zir-driver"

namespace cl = llvm::cl;
using namespace mlir;

static cl::OptionCategory ZirgenDriverCategory("Zklang zirgen options");

static cl::opt<std::string> inputFilename(
    cl::Positional, cl::desc("<input zirgen file>"), cl::value_desc("filename"), cl::Required,
    cl::cat(ZirgenDriverCategory)
);

namespace {
enum class Action {
  None = 0,
  PrintAST,
  PrintZHL,
  PrintZHLT,
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
        clEnumValN(Action::PrintZHLT, "zhlt", "Output typed high level ZIR IR"),
        clEnumValN(Action::PrintZML, "zml", "Output typed medium level ZIR IR"),
        clEnumValN(
            Action::OptimizeZML, "zmlopt",
            "Output typed medium level ZIR IR with separate compute and constrain functions"
        ),
        clEnumValN(Action::PrintLlzk, "llzk", "Output LLZK IR")
    ),
    cl::cat(ZirgenDriverCategory)
);

static cl::list<std::string> includeDirs(
    "I", cl::desc("Add include path"), cl::value_desc("path"), cl::cat(ZirgenDriverCategory)
);
static cl::opt<std::string> outputFile(
    "o", cl::desc("Where to write the result"), cl::value_desc("output"),
    cl::cat(ZirgenDriverCategory)
);
static cl::opt<bool> emitBytecode(
    "emit-bytecode", cl::desc("Emit IR in bytecode format"), cl::cat(ZirgenDriverCategory)
);
static cl::opt<bool> stripDebugInfo(
    "strip-debug-info", cl::desc("Toggle stripping debug information when writing the output"),
    cl::cat(ZirgenDriverCategory)
);

// Dev-oriented command line options only available in debug builds
// They are defined with external storage to avoid having to wrap
// the code that uses them in preprocessor checks too
bool DisableMultiThreadingFlag = false;
bool DisableCleanupPassesFlag = false;
bool DisableCastReconciliationFlag = false;

#ifndef NDEBUG
static cl::opt<bool, true> DisableMultiThreading(
    "disable-multithreading", cl::desc("Disable multithreading of the lowering pipeline"),
    cl::Hidden, cl::location(DisableMultiThreadingFlag), cl::cat(ZirgenDriverCategory)
);

static cl::opt<bool, true> DisableCleanupPasses(
    "disable-cleanup-passes", cl::desc("Disables running cse and canonicalize in the pipeline"),
    cl::Hidden, cl::location(DisableCleanupPassesFlag), cl::cat(ZirgenDriverCategory)
);

static cl::opt<bool, true> DisableCastReconciliation(
    "disable-reconciliation-passes",
    cl::desc("Disables running reconcile-unrealized-casts in the pipeline"), cl::Hidden,
    cl::location(DisableCastReconciliationFlag), cl::cat(ZirgenDriverCategory)
);
#endif

/// A wrapper around the dialect registry that ensures that the required dialects are available
/// at initialization.
struct ZirFrontendDialects {

  ZirFrontendDialects() {
    registry.insert<zml::ZMLDialect>();
    registry.insert<zirgen::Zhl::ZhlDialect>();
    llzk::registerAllDialects(registry);
  }
  mlir::DialectRegistry registry;
};

class Driver {
public:
  Driver(int &argc, char **&argv);
  Driver(const Driver &) = delete;
  Driver(Driver &&) = delete;
  Driver &operator=(const Driver &) = delete;
  Driver &operator=(Driver &&) = delete;
  LogicalResult run();

private:
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

Driver::Driver(int &argc, char **&argv)
    : llvm(argc, argv), dialects(), context(dialects.registry), sourceManager{},
      sourceMgrHandler(sourceManager, &context), pm(&context) {
#ifdef NDEBUG
  cl::HideUnrelatedOptions(ZirgenDriverCategory);
#endif
  cl::ParseCommandLineOptions(argc, argv, "zirgen frontend to LLZK\n");

  context.loadAllAvailableDialects();

  zml::loadLLZKDialectExtensions(context);
  if (DisableMultiThreadingFlag) {
    llvm::errs() << "Multithreading was disabled!\n";
    context.disableMultithreading();
  }

  pm.enableVerifier(true);
}
namespace {
/// Helper to ensure the pipeline manager is configured properly
class PMSetupGuard {
  mlir::PassManager &pm;

public:
  PMSetupGuard(mlir::PassManager &PM) : pm(PM) { pm.clear(); }

  ~PMSetupGuard() {
    if (stripDebugInfo) {
      pm.addPass(mlir::createStripDebugInfoPass());
    }
  }
};
} // namespace

void Driver::configureLoweringPipeline() {
  PMSetupGuard guard(pm);
  if (emitAction == Action::PrintZHL) {
    return;
  }

  pm.addPass(zklang::createInjectLlzkModAttrsPass());
  pm.addPass(zklang::createStripTestsPass());
  pm.addPass(zml::createInjectBuiltInsPass());
  // pm.addPass(zklang::createInstantiatePODBlocksPass());
  pm.addPass(zklang::createAnnotateTypecheckZhlPass());
  // TODO: Move this pass past Action::PrintZHLT
  pm.addPass(zklang::createConvertZhlToLlzkStructPass());
  if (emitAction == Action::PrintZHLT) {
    return;
  }
  pm.addPass(zklang::createConvertZhlToZmlPass());
  if (!DisableCastReconciliationFlag) {
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  }

  if (emitAction == Action::PrintZML) {
    return;
  }

  auto &compPm = pm.nest<zml::ComponentOp>();
  compPm.nest<mlir::func::FuncOp>().addPass(zml::createLowerBuiltInsPass());
  pm.addPass(zml::createRemoveBuiltInsPass());
  pm.addPass(zml::createSplitComponentBodyPass());
  auto &splitCompPipeline = pm.nest<zml::SplitComponentOp>();
  auto &splitCompFuncsPipeline = splitCompPipeline.nest<mlir::func::FuncOp>();
  splitCompFuncsPipeline.addPass(zml::createRemoveIllegalComputeOpsPass());
  splitCompFuncsPipeline.addPass(zml::createRemoveIllegalConstrainOpsPass());
  if (!DisableCleanupPassesFlag) {
    splitCompFuncsPipeline.addPass(mlir::createCSEPass());
  }

  if (emitAction == Action::OptimizeZML) {
    return;
  }
  pm.addPass(zklang::createConvertZmlToLlzkPass());
  auto &llzkStructPipeline = pm.nest<llzk::component::StructDefOp>();
  if (!DisableCastReconciliationFlag) {
    llzkStructPipeline.addPass(mlir::createReconcileUnrealizedCastsPass());
  }
  if (!DisableCleanupPassesFlag) {
    llzkStructPipeline.addPass(mlir::createCanonicalizerPass());
  }
}

zirgen::dsl::ast::Module::Ptr Driver::parse() {
  zirgen::dsl::Parser parser(sourceManager);
  parser.addPreamble(zml::zirPreamble);
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
      out = base + (emitBytecode ? ".bc" : ".mlir");
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

  void writeModule(mlir::ModuleOp mod) {
    if (emitBytecode) {
      mlir::BytecodeWriterConfig config;
      // This will include the debug info as well unless `--strip-debug-info`
      // is specified.
      if (mlir::writeBytecodeToFile(mod, *dst, config).failed()) {
        mod->emitOpError("could not write module bytecode to file").report();
      }
    } else {
      mod->print(*dst, OpPrintingFlags(std::nullopt).enableDebugInfo(!stripDebugInfo));
    }
  }

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
  configureLoweringPipeline();
  LLVM_DEBUG(pm.dump());
  if (failed(pm.run(*mod))) {
    DEBUG_WITH_TYPE("zir-driver-dump-on-error", llvm::errs() << "Module contents:\n";
                    mod->print(llvm::errs(), OpPrintingFlags(std::nullopt).printGenericOpForm()));
    llvm::WithColor(llvm::errs(), llvm::raw_ostream::RED, true)
        << "An internal compiler error ocurred while lowering this module.\n";
    return failure();
  }

  dst.writeModule(*mod);

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

  Driver driver(argc, argv);
  return driver.run();
}

} // namespace zklang

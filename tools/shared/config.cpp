#include "version.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llzk/Config/Config.h>
#include <tools/config.h>

#define BUG_REPORT_URL "https://github.com/Veridise/zirgen-to-llzk/issues"

void zklang::configureTool() {
  llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());
  llvm::setBugReportMsg("PLEASE submit a bug report to " BUG_REPORT_URL
                        " and include the crash backtrace, relevant input files,"
                        " and associated run script(s).\n");
  llvm::cl::AddExtraVersionPrinter([](llvm::raw_ostream &os) {
    os << "\nLLZK (" LLZK_URL "):\n  LLZK version " LLZK_VERSION_STRING "\n";
    os << "\nzklang (" ZKLANG_URL "):\n  zklang version " ZKLANG_VERSION_STRING "\n";
  });
}

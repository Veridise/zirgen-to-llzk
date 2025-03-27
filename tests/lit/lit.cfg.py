import os

import lit.llvm
from lit import formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "zklang"

# Disable container overflow checks because it can give false positives in
# ConvertZmlToLlzkPass::runOnOperation() since LLVM itself is not built with ASan.
# https://github.com/google/sanitizers/wiki/AddressSanitizerContainerOverflow#false-positives
config.environment["ASAN_OPTIONS"] = "detect_container_overflow=0:detect_leaks=0"

# Configuration file for the 'lit' test runner.
config.test_format = formats.ShTest(True)

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = [".mlir", ".zir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.zklang_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.zklang_obj_root, "test")
config.zklang_tools_dir = os.path.join(config.zklang_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.zklang_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.zklang_tools_dir, config.llvm_tools_dir]
tools = ["zklang-opt", "zklang"]

llvm_config.add_tool_substitutions(tools, tool_dirs)

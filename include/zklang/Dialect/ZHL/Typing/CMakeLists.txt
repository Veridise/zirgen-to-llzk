include_directories(SYSTEM ${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Passes.td")
mlir_tablegen(Passes.h.inc -gen-pass-decls)
zklang_add_mlir_doc(ZhlTypingPassesDocGen passes/ZhlTypingPasses.md -gen-pass-doc)

add_public_tablegen_target(ZhlTypingPassesIncGen)
add_dependencies(mlir-headers ZhlTypingPassesIncGen)
add_dependencies(ZklangHeaders ZhlTypingPassesIncGen)

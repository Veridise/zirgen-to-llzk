include_directories(SYSTEM ${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Passes.td")
mlir_tablegen(Passes.h.inc -gen-pass-decls)
zklang_add_mlir_doc(PassesDocGen passes/Passes.md -gen-pass-doc)

add_public_tablegen_target(PassesIncGen)
add_dependencies(mlir-headers PassesIncGen)
add_dependencies(ZklangHeaders PassesIncGen)

include_directories(SYSTEM ${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Passes.td")
mlir_tablegen(Passes.h.inc -gen-pass-decls)
zklang_add_mlir_doc(ZmlTransformsPassesDocGen passes/ZmlTransformationPasses.md -gen-pass-doc)

add_public_tablegen_target(ZmlTransformsPassesIncGen)
add_dependencies(mlir-headers ZmlTransformsPassesIncGen)
add_dependencies(ZklangHeaders ZmlTransformsPassesIncGen)

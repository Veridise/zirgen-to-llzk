include_directories(SYSTEM ${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Passes.td")
mlir_tablegen(Passes.h.inc -gen-pass-decls )
# llzk_add_mlir_doc(LLZKTransformationPassesDocGen LLZKTransformationPasses.md -gen-pass-doc)

add_public_tablegen_target(ZhlTypingPassesIncGen)
add_dependencies(mlir-headers ZhlTypingPassesIncGen)
add_dependencies(ZklangHeaders ZhlTypingPassesIncGen)

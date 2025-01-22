add_executable(zklang-opt main.cpp)

target_link_libraries(zklang-opt PUBLIC ${LLZK_DEP_DIALECT_LIBS} MLIROptLib
                                      LLZK::DialectRegistration ZIRC)
# llzk_target_add_mlir_link_settings(llzk-opt)
target_link_libraries(zklang-opt PUBLIC ZklangHeaders LLZKDialectHeaders MLIRIR)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

set_target_properties(zklang-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${CMAKE_BINARY_DIR}/bin)
# install(
#   TARGETS llzk-opt
#   EXPORT LLZKTargets
#   COMPONENT devtools)

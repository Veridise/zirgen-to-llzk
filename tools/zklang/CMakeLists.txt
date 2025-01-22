add_executable(zklang main.cpp)

target_link_libraries(zklang PUBLIC ${LLZK_DEP_DIALECT_LIBS} 
                                      LLZK::DialectRegistration ZIRC ZirFrontend)
# llzk_target_add_mlir_link_settings(llzk-opt)
target_link_libraries(zklang PUBLIC ZklangHeaders LLZK::LLZKDialectHeaders MLIRIR)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

set_target_properties(zklang PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${CMAKE_BINARY_DIR}/bin)
install(
  TARGETS zklang
  EXPORT ZklangTargets
  COMPONENT frontend)

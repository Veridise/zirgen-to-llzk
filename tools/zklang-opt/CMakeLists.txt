add_executable(zklang-opt main.cpp)

target_link_libraries(zklang-opt PUBLIC ${LLZK_DEP_DIALECT_LIBS} MLIROptLib
                                      LLZK::DialectRegistration ZIRC ZHLDialectHeaders)
# llzk_target_add_mlir_link_settings(llzk-opt)
target_link_libraries(zklang-opt PUBLIC   ZklangHeaders 
  LLZK::LLZKDialectHeaders 
  MLIRIR 
  MLIRTransforms 
  MLIRFuncDialect 
  MLIRIndexDialect 
  LLZK::LLZKDialect 
  ZIRParser 
  ZIRC
  MLIRReconcileUnrealizedCasts 
  MLIRPass)
target_link_libraries(zklang-opt PRIVATE zklang_options zklang_warnings)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

set_target_properties(zklang-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${CMAKE_BINARY_DIR}/bin)
install(
  TARGETS zklang-opt
  EXPORT ZklangTargets
  COMPONENT devtools)

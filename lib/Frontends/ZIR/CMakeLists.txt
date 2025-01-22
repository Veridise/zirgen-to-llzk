add_library(ZirFrontend Driver.cpp)

target_link_libraries(ZirFrontend PUBLIC ${LLZK_DEP_DIALECT_LIBS} 
                                      LLZK::DialectRegistration ZIRC)
# llzk_target_add_mlir_link_settings(llzk-opt)
target_link_libraries(ZirFrontend PUBLIC ZklangHeaders LLZKDialectHeaders MLIRIR)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})



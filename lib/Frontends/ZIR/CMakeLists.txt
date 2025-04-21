add_library(ZirFrontend Driver.cpp)

target_link_libraries(ZirFrontend PUBLIC ${LLZK_DEP_DIALECT_LIBS}
                                      LLZK::DialectRegistration ZIRC ZHLDialectHeaders)

target_link_libraries(ZirFrontend PUBLIC
  ZklangHeaders
  LLZK::LLZKDialectHeaders
  MLIRIR
  MLIRTransforms
  MLIRFuncDialect
  MLIRIndexDialect
  LLZK::LLZKDialect
  ZIRParser
  ZIRC
  MLIRReconcileUnrealizedCasts
  MLIRDebug
  MLIRBytecodeWriter
  MLIRPass)
target_link_libraries(ZirFrontend PRIVATE zklang_options zklang_warnings)
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

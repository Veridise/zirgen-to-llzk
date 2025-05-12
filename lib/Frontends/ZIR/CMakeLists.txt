add_library(ZirFrontend Driver.cpp)

target_link_libraries(
  ZirFrontend PUBLIC ${LLZK_DEP_DIALECT_LIBS} LLZK::DialectRegistration
                     ZHLDialectHeaders)

target_link_libraries(
  ZirFrontend
  PUBLIC ZklangHeaders
         LLZK::LLZKDialectHeaders
         MLIRIR
         MLIRTransforms
         MLIRFuncDialect
         MLIRIndexDialect
         LLZK::DialectRegistration
         LLZK::AllDialects
         LLZKUtil
         ZIRParser
         ZklangPasses
         ZMLDialectTransforms
         ZMLDialect
         MLIRReconcileUnrealizedCasts
         MLIRDebug
         MLIRBytecodeWriter
         MLIRPass)
target_link_libraries(ZirFrontend PRIVATE zklang_options zklang_warnings)
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

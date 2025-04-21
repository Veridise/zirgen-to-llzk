add_library(ZIRC)

target_link_libraries(ZIRC INTERFACE LLVMHeaders MLIRHeaders)
target_link_libraries(
  ZIRC
  PRIVATE ZHLDialectHeaders
          ZHLDialect
          ZklangHeaders
          MLIRSupport
          MLIRTransformUtils
          LLZK::LLZKDialect
          MLIRSCFTransforms)
target_link_libraries(ZIRC PRIVATE zklang_options zklang_warnings)
target_include_directories(ZIRC PUBLIC ${ZKLANG_INCLUDE_DIR})
add_dependencies(ZIRC ZHLDialect ZklangHeaders)

add_subdirectory(Dialect)
add_subdirectory(Passes)
add_subdirectory(Frontends)

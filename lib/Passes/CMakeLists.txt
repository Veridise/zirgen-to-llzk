add_library(ZklangPasses)

target_link_libraries(ZklangPasses INTERFACE LLVMHeaders MLIRHeaders)
target_link_libraries(
  ZklangPasses
  PRIVATE ZHLDialectHeaders
          ZHLDialect
          ZklangHeaders
          MLIRSupport
          MLIRTransformUtils
          ZMLDialect
          MLIRIndexDialect
          LLZK::AllDialects
          LLZKUtil
          MLIRSCFTransforms)
target_link_libraries(ZklangPasses PRIVATE zklang_options zklang_warnings)
target_include_directories(ZklangPasses PUBLIC ${ZKLANG_INCLUDE_DIR})
add_dependencies(ZklangPasses ZHLDialect ZklangHeaders)

target_sources(ZklangPasses PRIVATE StripTests.cpp)

add_subdirectory(ConvertZhlToZml)
add_subdirectory(ConvertZmlToLlzk)

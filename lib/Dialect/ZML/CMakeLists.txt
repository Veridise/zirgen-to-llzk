add_library(ZMLDialect)

target_link_libraries(ZMLDialect INTERFACE LLVMHeaders MLIRHeaders)
target_link_libraries(
  ZMLDialect
  PRIVATE ZHLDialectHeaders
          ZHLDialect
          ZklangHeaders
          MLIRSupport
          ZHLTyping
          MLIRIndexDialect
          LLZK::LLZKDialectHeaders
          LLZK::LLZKStructDialect
          LLZK::LLZKFunctionDialect
          LLZK::LLZKArrayDialect
          LLZK::LLZKUndefDialect
          LLZK::LLZKDialect
          LLZKUtil
          LLZK::LLZKGlobalDialect
          LLZK::LLZKPolymorphicDialect
          LLZK::LLZKFeltDialect
          LLZK::LLZKStringDialect
          MLIRTransformUtils
          MLIRSCFTransforms)
target_link_libraries(ZMLDialect PRIVATE zklang_options zklang_warnings)
target_include_directories(ZMLDialect PUBLIC ${ZKLANG_INCLUDE_DIR})
add_dependencies(ZMLDialect ZHLDialect ZklangHeaders)

add_subdirectory(BuiltIns)
add_subdirectory(IR)
add_subdirectory(Typing)
add_subdirectory(Transforms)
add_subdirectory(ExtVal)

target_sources(ZMLDialect PRIVATE Helpers.cpp)

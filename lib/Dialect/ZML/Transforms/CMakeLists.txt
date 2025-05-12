add_library(ZMLDialectTransforms
InjectBuiltIns.cpp 
  LowerBuiltIns.cpp 
  RemoveBuiltIns.cpp 
  RemoveIllegalOps.cpp 
  SplitComponentBody.cpp
)

target_link_libraries(ZMLDialectTransforms INTERFACE LLVMHeaders MLIRHeaders)
target_link_libraries(
  ZMLDialectTransforms
  PRIVATE ZHLDialectHeaders
          ZHLDialect
          ZklangHeaders
          MLIRSupport
          ZHLTyping
          MLIRIndexDialect
          LLZKTyping
          ZMLDialect
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
target_link_libraries(ZMLDialectTransforms PRIVATE zklang_options zklang_warnings)
target_include_directories(ZMLDialectTransforms PUBLIC ${ZKLANG_INCLUDE_DIR})
add_dependencies(ZMLDialectTransforms ZHLDialect ZklangHeaders)




add_library(LLZKTyping)

target_link_libraries(LLZKTyping INTERFACE LLVMHeaders MLIRHeaders)
target_link_libraries(LLZKTyping PRIVATE  ZMLDialect
                                        ZklangHeaders MLIRSupport)
target_link_libraries(LLZKTyping PRIVATE zklang_options zklang_warnings)
target_include_directories(LLZKTyping PUBLIC ${ZKLANG_INCLUDE_DIR})
add_dependencies(LLZKTyping ZMLDialect ZklangHeaders)

target_sources(
  LLZKTyping
  PRIVATE LLZKTypeConverter.cpp
Builder.cpp)

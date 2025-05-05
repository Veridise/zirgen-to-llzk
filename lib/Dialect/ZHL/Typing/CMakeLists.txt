add_library(ZHLTyping)

target_link_libraries(ZHLTyping INTERFACE LLVMHeaders MLIRHeaders)
target_link_libraries(
  ZHLTyping
  PRIVATE ZHLDialectHeaders ZHLDialect ZklangHeaders MLIRSupport
          # MLIRTransformUtils LLZK::AllDialects LLZKUtil MLIRSCFTransforms
)
target_link_libraries(ZHLTyping PRIVATE zklang_options zklang_warnings)
target_include_directories(ZHLTyping PUBLIC ${ZKLANG_INCLUDE_DIR})
add_dependencies(ZHLTyping ZHLDialect ZklangHeaders)

target_sources(
  ZHLTyping
  PRIVATE Analysis.cpp
          OpBindings.cpp
          Rules.cpp
          Scope.cpp
          TypeBindings.cpp
          TypeBindingSpecialization.cpp
          Typing.cpp
          Frame.cpp
          Expr.cpp
          TypeBinding.cpp
          Params.cpp
          Interpreter.cpp)

add_library(ZHLTypingPasses)

target_link_libraries(ZHLTypingPasses INTERFACE LLVMHeaders MLIRHeaders)
target_link_libraries(
  ZHLTypingPasses
  PRIVATE ZHLDialectHeaders
          # ZHLDialect
          ZHLTyping ZklangHeaders MLIRSupport
  # MLIRTransformUtils LLZK::AllDialects LLZKUtil MLIRSCFTransforms
)
target_link_libraries(ZHLTypingPasses PRIVATE zklang_options zklang_warnings)
target_include_directories(ZHLTypingPasses PUBLIC ${ZKLANG_INCLUDE_DIR})
add_dependencies(ZHLTypingPasses ZHLDialect ZklangHeaders)

target_sources(ZHLTypingPasses PRIVATE PrintTypeBindings.cpp)

include_directories(SYSTEM ${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})
include_directories(${ZKLANG_INCLUDE_DIR})
add_executable(ZHLTests TypingTests.cpp)

target_link_libraries(
  ZHLTests PUBLIC ${GTEST_LIB_TARGETS} ${GTEST_EXE_TARGET} ZHLTyping
                  ZklangHeaders ZMLDialect LLVMSupport)
target_link_libraries(ZHLTests PRIVATE zklang_options zklang_warnings)

zklang_gtest_suite(ZHLTests)

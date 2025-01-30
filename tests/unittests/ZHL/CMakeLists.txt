# Simple hello world test that validates the GTest setup works

include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR} ${ZKLANG_INCLUDE_DIR})
add_executable(ZHLTests TypingTests.cpp)

target_link_libraries(
  ZHLTests PUBLIC ZIRC ${GTEST_LIB_TARGETS}
                        ${GTEST_EXE_TARGET} ZklangHeaders MLIRParser LLVMSupport)

zklang_gtest_suite(ZHLTests)

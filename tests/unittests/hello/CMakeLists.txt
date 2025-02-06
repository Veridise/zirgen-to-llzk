# Simple hello world test that validates the GTest setup works

add_executable(HelloWorldTest HelloWorldTest.cpp)

target_link_libraries(
  HelloWorldTest PUBLIC ${GTEST_LIB_TARGETS}
                        ${GTEST_EXE_TARGET}  MLIRParser)

zklang_gtest_suite(HelloWorldTest)

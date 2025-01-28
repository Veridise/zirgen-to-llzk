function(zklang_gtest_suite target)
  gtest_discover_tests(${target})
  list(APPEND ZKLANG_UNITTESTS ${target})
  set(ZKLANG_UNITTESTS
      ${ZKLANG_UNITTESTS}
      CACHE INTERNAL "")
endfunction()

add_subdirectory(helloworld) # example

add_custom_target(
  check-unit
  DEPENDS ${ZKLANG_UNITTESTS}
  COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target test
  COMMENT "Run the gtest-based unit tests")

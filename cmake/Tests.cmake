# Inspired by https://github.com/cpp-best-practices/cmake_template/blob/main/cmake/Tests.cmake

function(zklang_enable_coverage project_name)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    target_compile_options(${project_name} INTERFACE --coverage -O0 -g)
    target_link_libraries(${project_name} INTERFACE --coverage)
  endif()
endfunction()

macro(zklang_setup_testing)
  include(CTest)
  enable_testing()
  set(CMAKE_CTEST_ARGUMENTS
      "--output-on-failure"
      CACHE STRING "CTest arguments")
  if(BUILD_TESTING)
    add_subdirectory(tests)
  endif()

  # Catch-all target for running all tests
  add_custom_target(
    check
    DEPENDS # test targets may not exist if BUILD_TESTING is off
          $<$<BOOL:BUILD_TESTING>:check-unit>
          $<$<BOOL:BUILD_TESTING>:check-lit>
  )
endmacro()

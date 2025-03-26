# Inspired by https://github.com/cpp-best-practices/cmake_template/blob/main/ProjectOptions.cmake

include(CMakeDependentOption)
include(CheckCXXCompilerFlag)

macro(zklang_supports_sanitizers)
  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND NOT WIN32)
    set(SUPPORTS_UBSAN ON)
  else()
    set(SUPPORTS_UBSAN OFF)
  endif()

  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND WIN32)
    set(SUPPORTS_ASAN OFF)
  else()
    set(SUPPORTS_ASAN ON)
  endif()
endmacro()

macro(zklang_setup_options)
  option(ZKLANG_ENABLE_HARDENING "Enable hardening" ON)
  option(ZKLANG_ENABLE_COVERAGE "Enable coverage reporting" OFF)
  cmake_dependent_option(
    ZKLANG_ENABLE_GLOBAL_HARDENING
    "Attempt to push hardening options to built dependencies"
    ON
    ZKLANG_ENABLE_HARDENING
    OFF)

  zklang_supports_sanitizers()

  option(ZKLANG_WARNINGS_AS_ERRORS "Treat Warnings As Errors" ON)
  option(ZKLANG_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" ${SUPPORTS_ASAN})
  option(ZKLANG_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
  option(ZKLANG_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" ${SUPPORTS_UBSAN})
  option(ZKLANG_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
  option(ZKLANG_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
endmacro()

macro(zklang_global_options)
  zklang_supports_sanitizers()

  if(ZKLANG_ENABLE_HARDENING AND ZKLANG_ENABLE_GLOBAL_HARDENING)
    include(cmake/Hardening.cmake)
    if(NOT SUPPORTS_UBSAN 
       OR ZKLANG_ENABLE_SANITIZER_UNDEFINED
       OR ZKLANG_ENABLE_SANITIZER_ADDRESS
       OR ZKLANG_ENABLE_SANITIZER_THREAD
       OR ZKLANG_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()
    message("${ZKLANG_ENABLE_HARDENING} ${ENABLE_UBSAN_MINIMAL_RUNTIME} ${ZKLANG_ENABLE_SANITIZER_UNDEFINED}")
    zklang_enable_hardening(ZKLANG_options ON ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()
endmacro()

macro(zklang_local_options)
  if(PROJECT_IS_TOP_LEVEL)
    include(cmake/StandardProjectSettings.cmake)
  endif()

  add_library(zklang_warnings INTERFACE)
  add_library(zklang_options INTERFACE)

  include(cmake/CompilerWarnings.cmake)
  zklang_set_project_warnings(
    zklang_warnings
    ${ZKLANG_WARNINGS_AS_ERRORS}
    ""
    "")

  include(cmake/Sanitizers.cmake)
  zklang_enable_sanitizers(
    zklang_options
    ${ZKLANG_ENABLE_SANITIZER_ADDRESS}
    ${ZKLANG_ENABLE_SANITIZER_LEAK}
    ${ZKLANG_ENABLE_SANITIZER_UNDEFINED}
    ${ZKLANG_ENABLE_SANITIZER_THREAD}
    ${ZKLANG_ENABLE_SANITIZER_MEMORY})

  if(ZKLANG_ENABLE_COVERAGE)
    include(cmake/Tests.cmake)
    zklang_enable_coverage(zklang_options)
  endif()

  if(ZKLANG_ENABLE_HARDENING AND NOT ZKLANG_ENABLE_GLOBAL_HARDENING)
    include(cmake/Hardening.cmake)
    if(NOT SUPPORTS_UBSAN 
       OR ZKLANG_ENABLE_SANITIZER_UNDEFINED
       OR ZKLANG_ENABLE_SANITIZER_ADDRESS
       OR ZKLANG_ENABLE_SANITIZER_THREAD
       OR ZKLANG_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()
    zklang_enable_hardening(zklang_options OFF ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()

endmacro()

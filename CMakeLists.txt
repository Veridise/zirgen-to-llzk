cmake_minimum_required(VERSION 3.23)

project(
  Zklang
  DESCRIPTION
    "Veridise's ZIR frontend for LLZK"
  VERSION 0.1.0
  HOMEPAGE_URL https://github.com/Veridise/zir-to-zkir) # TODO: Change the repo name

option(ZKLANG_TEMPLATE_BACKTRACE_LIMIT
       "Add -ftemplate-backtrace-limit=0 to the compile flags")

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD
    20
    CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED
    ON
    CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Use same policy as LLVM to suppress warnings
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")

# Dependency setup

message(STATUS "${CMAKE_MODULE_PATH}")
find_package(LLVM 18.1 REQUIRED CONFIG)
message(STATUS "Using LLVM in: ${LLVM_DIR}")
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIR in: ${MLIR_DIR}")
find_package(LLZK REQUIRED CONFIG)
message(STATUS "Using LLZK in: ${LLZK_DIR}")

# Adds an INTERFACE library that only includes headers
function(add_header_library _name _include_dirs _export_name)
  add_library(${_name} INTERFACE)
  target_include_directories(${_name} INTERFACE ${_include_dirs})
  install(TARGETS ${_name} EXPORT ${_export_name})
endfunction()

set(ZKLANG_EXPORT_TARGETS "ZklangTargets")
# LLVM & MLIR do not propagate their include dirs correctly We define them as
# INTERFACE libraries and link against them
add_header_library(LLVMHeaders ${LLVM_INCLUDE_DIRS} ${ZKLANG_EXPORT_TARGETS})
add_header_library(MLIRHeaders ${MLIR_INCLUDE_DIRS} ${ZKLANG_EXPORT_TARGETS})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
add_compile_options(
  -Werror=switch
  -Werror=return-type
  -Wall
  -Wextra
  -Wno-unused-parameter
  -Wshadow
  -Wnon-virtual-dtor
  -pedantic
  -fno-omit-frame-pointer)

if(CMAKE_CONFIGURATION_TYPES)
  list(APPEND CMAKE_CONFIGURATION_TYPES DebWithSans)
  list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
endif()

set(ZKLANG_LIT_COMPATIBLE_COVERAGE
    OFF
    CACHE INTERNAL "")

if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DebWithSans")
  if(ZKLANG_LIT_COMPATIBLE_COVERAGE)
    message(FATAL_ERROR "Don't manually set ZKLANG_LIT_COMPATIBLE_COVERAGE.")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    message(STATUS "Adding coverage flags")
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
    set(ZKLANG_LIT_COMPATIBLE_COVERAGE ON)
    message(
      STATUS "ZKLANG_LIT_COMPATIBLE_COVERAGE=${ZKLANG_LIT_COMPATIBLE_COVERAGE}")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(WARNING "Gcov reports are not supported by the test environment.")
    message(WARNING "Set ZKLANG_FORCE_GCC_COVERAGE to enable coverage in GCC.")
    if(ZKLANG_FORCE_GCC_COVERAGE)
      add_compile_options(--coverage -g)
      add_link_options(--coverage)
    endif()
  else()
    message(
      WARNING
        "Coverage requested but ${CMAKE_CXX_COMPILER_ID} is not a supported compiler and coverage will not be applied."
    )
  endif()

  add_compile_options(-g)
endif()


# Temporary arrangement for enabling ASAN on the "debug" nix derivations
if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  # NOTE: undefined behavior sanitizer ("undefined") has been removed due to its
  # poor interaction with gtest fixtures and other standard MLIR/LLVM
  # operations.
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # NOTE: undefined behavior sanitizer ("undefined") has been removed due to its
  # poor interaction with gtest fixtures and other standard MLIR/LLVM
  # operations.
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

if(ZKLANG_TEMPLATE_BACKTRACE_LIMIT)
  add_compile_options(-ftemplate-backtrace-limit=0)
endif()




# Project files

set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc")
set(ZKLANG_MLIR_DOC_OUTPUT_DIR "${DOXYGEN_OUTPUT_DIRECTORY}/mlir")

# We need the include dir so we can pass it to mlir-tblgen
set(ZKLANG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ZKLANGMacros.cmake)

# Create an empty list for the dialect libraries, to be populated in the lib
# subdirectory.
# add_library(LLZKAllDialects INTERFACE)
# add_library(LLZK::AllDialects ALIAS LLZKAllDialects)

# Ensures that the target can be imported as "LLZK::AllDialects" and not
# "LLZK::LLZKAllDialects"
# set_target_properties(LLZKAllDialects PROPERTIES EXPORT_NAME "AllDialects")
# install(TARGETS LLZKAllDialects EXPORT LLZKTargets)

add_subdirectory(third-party)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# Documentation
# add_custom_target(doc)
# find_package(Doxygen OPTIONAL_COMPONENTS dot)
# if(Doxygen_FOUND)
#   message(STATUS "Doxygen found, enabling documentation...")
#   set(DOXYGEN_EXTRACT_ALL YES)
#   set(DOXYGEN_INCLUDE_PATH "${CMAKE_CURRENT_BINARY_DIR}/include/")
#   set(DOXYGEN_EXCLUDE_PATTERNS ${CMAKE_CURRENT_BINARY_DIR}/include/*/*.md)
#   set(DOXYGEN_USE_MDFILE_AS_MAINPAGE
#       ${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/index.md)
#   set(DOXYGEN_FILE_PATTERNS
#       *.cpp
#       *.cpp.inc
#       *.h.inc
#       *.hpp
#       *.h
#       *.td
#       *.md
#       *.py)
#   set(DOXYGEN_EXTENSION_MAPPING inc=C++)
#   set(DOXYGEN_MACRO_EXPANSION YES)
#   set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
#   set(DOXYGEN_PREDEFINED GET_OP_CLASSES GET_TYPEDEF_CLASSES GET_ATTR_CLASSES)
#   set(DOXYGEN_SOURCE_BROWSER YES)
#   set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
#   doxygen_add_docs(
#     doxygen
#     "${CMAKE_CURRENT_SOURCE_DIR}/lib/"
#     "${CMAKE_CURRENT_SOURCE_DIR}/tools/"
#     "${CMAKE_CURRENT_BINARY_DIR}/include/"
#     "${CMAKE_CURRENT_SOURCE_DIR}/include/"
#     "${ZKLANG_MLIR_DOC_OUTPUT_DIR}"
#     "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/")
#   add_dependencies(doxygen ZKLANGDialectHeaders LLZKDialectHeaders mlir-doc)
#   add_dependencies(doc doxygen)
# endif()

# Tests

# lit tests
include(CTest)
enable_testing()
# set(CMAKE_CTEST_ARGUMENTS
#     "--output-on-failure"
#     CACHE STRING "CTest arguments")
# if(BUILD_TESTING)
#   add_subdirectory(test)
# endif()
#
find_package(GTest)
if(BUILD_TESTING AND GTest_FOUND)
  message(STATUS "gtest found, enabling unit tests...")
  include(GoogleTest)

  # add_custom_target(LLZKUnitTests)

  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20")
    set(GTEST_LIB_TARGETS "GTest::gtest;GTest::gmock")
    set(GTEST_EXE_TARGET GTest::gtest_main)
  else()
    set(GTEST_LIB_TARGETS GTest::GTest)
    set(GTEST_EXE_TARGET GTest::Main)
  endif()

  add_subdirectory(tests/unittests)
elseif(BUILD_TESTING)
  message(STATUS "gtest not found, unit tests will not be run")
else()
  message(STATUS "Unit tests are disabled")
endif()
#
# # Catch-all target for running all tests
# add_custom_target(
#   check
#   DEPENDS # test targets may not exist if BUILD_TESTING is off
#           $<$<BOOL:BUILD_TESTING>:check-unit>
#           $<$<BOOL:BUILD_TESTING>:check-lit>)
#
# # Install

export(
  EXPORT ${ZKLANG_EXPORT_TARGETS}
  FILE ${CMAKE_CURRENT_BINARY_DIR}/${ZKLANG_EXPORT_TARGETS}.cmake
  NAMESPACE ZKLANG::)
install(
  EXPORT ${ZKLANG_EXPORT_TARGETS}
  NAMESPACE ZKLANG::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/zklang)

# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
#   "${CMAKE_CURRENT_BINARY_DIR}/ZklangConfigVersion.cmake"
#   VERSION "${PROJECT_VERSION}"
#   COMPATIBILITY AnyNewerVersion)
# configure_package_config_file(
#   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ZklangConfig.cmake.in"
#   "${CMAKE_CURRENT_BINARY_DIR}/ZklangConfig.cmake"
#   INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/zklang")
# install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ZklangConfig.cmake"
#               "${CMAKE_CURRENT_BINARY_DIR}/ZklangConfigVersion.cmake"
#         DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Zklang")

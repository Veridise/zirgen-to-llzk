cmake_minimum_required(VERSION 3.23)

if (NOT DEFINED ZKLANG_VERSION_OVERRIDE)
  include(cmake/GitVersion.cmake)
  get_git_version(ZKLANG_VERSION "0.1.0") # default version
else()
  message(STATUS "ZKLANG_VERSION_OVERRIDE specified, setting version to ${ZKLANG_VERSION_OVERRIDE}")
  set(ZKLANG_VERSION "${ZKLANG_VERSION_OVERRIDE}")
endif()

project(
  Zklang
  DESCRIPTION
    "Veridise's Zirgen frontend for LLZK"
  VERSION "${ZKLANG_VERSION}"
  HOMEPAGE_URL https://github.com/Veridise/zirgen-to-llzk)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD
      20
      CACHE STRING "C++ standard")
endif()

include(GNUInstallDirs)
include(cmake/PreventInSourceBuilds.cmake)
include(ProjectOptions.cmake)

zklang_setup_options()
zklang_global_options()

set(ZKLANG_EXPORT_TARGETS "ZklangTargets")

include(Dependencies.cmake)
zklang_setup_dependencies(${ZKLANG_EXPORT_TARGETS})

zklang_local_options()

include(cmake/Docgen.cmake)

# Project files
add_subdirectory(third-party)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# Docs and doc targets setup
list(APPEND DOXYGEN_DEPS ZklangHeaders mlir-doc)
zklang_setup_doc_generation("${DOXYGEN_DEPS}")

include(cmake/Tests.cmake)
zklang_setup_testing()

# Install
include(cmake/Install.cmake)
zklang_setup_install_targets()

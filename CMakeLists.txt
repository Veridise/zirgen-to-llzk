cmake_minimum_required(VERSION 3.23)

project(
  Zklang
  DESCRIPTION "Veridise's ZIR frontend for LLZK"
  VERSION 0.1.0
  HOMEPAGE_URL https://github.com/Veridise/zir-to-zkir) # TODO: Change the repo
                                                        # name

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

# Project files
add_subdirectory(third-party)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

include(cmake/Docgen.cmake)
list(APPEND DOXYGEN_DEPS ZklangHeaders LLZKDialectHeaders mlir-doc)
zklang_setup_doc_generation(${DOXYGEN_DEPS})

include(cmake/Tests.cmake)
zklang_setup_testing()

# Install
include(cmake/Install.cmake)
zklang_setup_install_targets()

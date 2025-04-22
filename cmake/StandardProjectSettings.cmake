# Inspired by https://github.com/cpp-best-practices/cmake_template/blob/main/cmake/StandardProjectSettings.cmake

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE
      RelWithDebInfo
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui, ccmake
  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS
             "Debug"
             "Release"
             "MinSizeRel"
             "RelWithDebInfo")
endif()

# Generate compile_commands.json to make it easier to work with clang based tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(ZKLANG_ENABLE_COLOR_DIAGNOSTICS)
  # Enhance error reporting and compiler messages
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*")
    add_compile_options(-fcolor-diagnostics)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*")
    add_compile_options(-fdiagnostics-color=auto)
  else()
    message(STATUS "No colored compiler diagnostic set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
  endif()
endif()

# We need the include dir so we can pass it to mlir-tblgen
set(ZKLANG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

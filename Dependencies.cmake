# Adds an INTERFACE library that only includes headers
function(add_header_library _name _include_dirs _export_name)
  add_library(${_name} INTERFACE)
  target_include_directories(${_name} INTERFACE ${_include_dirs})
  install(TARGETS ${_name} EXPORT ${_export_name})
endfunction()

macro(zklang_setup_dependencies ZKLANG_EXPORT_TARGETS)

  # Use same policy as LLVM to suppress warnings
  if(POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
  endif()


  # Dependency setup

  message(DEBUG "${CMAKE_MODULE_PATH}")

  find_package(LLVM 18.1 REQUIRED CONFIG)
  message(STATUS "Using LLVM in: ${LLVM_DIR}")
  
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIR in: ${MLIR_DIR}")
  
  find_package(LLZK REQUIRED CONFIG)
  message(STATUS "Using LLZK in: ${LLZK_DIR}")

  # LLVM & MLIR do not propagate their include dirs correctly, so we define them as
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
endmacro()

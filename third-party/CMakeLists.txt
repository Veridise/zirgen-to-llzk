# Cmake targets for the third party components obtained via the submodules in the same directory as this file.

############## zirgen's ZHL Dialect ##############

# Header only library for users of the ZHL dialect
add_library(ZHLDialectHeaders INTERFACE)
target_include_directories(
  ZHLDialectHeaders SYSTEM
  INTERFACE # Headers in the ZHL IR folder
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/zirgen/zirgen/Dialect/ZHL/IR>
            # For the includes from the top of the repository
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/zirgen/>
            # TableGen'd files
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/zirgen/Dialect/ZHL/IR>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/zirgen/Dialect/ZHL/IR>)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/zirgen/zirgen/Dialect/ZHL/IR" "${CMAKE_CURRENT_BINARY_DIR}/zirgen/Dialect/ZHL/IR"
  TYPE INCLUDE
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.h.inc"
  PATTERN "*.td"
  PATTERN CMakeFiles EXCLUDE)

# TableGen's files for the ZHL dialect
include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/zirgen/Dialect/ZHL/IR")

set(LLVM_TARGET_DEFINITIONS "${CMAKE_CURRENT_SOURCE_DIR}/zirgen/zirgen/Dialect/ZHL/IR/Dialect.td")
mlir_tablegen(zirgen/Dialect/ZHL/IR/Dialect.h.inc -gen-dialect-decls -dialect=zhl -I${CMAKE_CURRENT_SOURCE_DIR}/zirgen)
mlir_tablegen(zirgen/Dialect/ZHL/IR/Dialect.cpp.inc -gen-dialect-defs -dialect=zhl -I${CMAKE_CURRENT_SOURCE_DIR}/zirgen)

set(LLVM_TARGET_DEFINITIONS "${CMAKE_CURRENT_SOURCE_DIR}/zirgen/zirgen/Dialect/ZHL/IR/Attrs.td")
mlir_tablegen(zirgen/Dialect/ZHL/IR/Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=zhl -I${CMAKE_CURRENT_SOURCE_DIR}/zirgen)
mlir_tablegen(zirgen/Dialect/ZHL/IR/Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=zhl -I${CMAKE_CURRENT_SOURCE_DIR}/zirgen)

set(LLVM_TARGET_DEFINITIONS "${CMAKE_CURRENT_SOURCE_DIR}/zirgen/zirgen/Dialect/ZHL/IR/Types.td")
mlir_tablegen(zirgen/Dialect/ZHL/IR/Types.h.inc -gen-typedef-decls -typedefs-dialect=zhl -I${CMAKE_CURRENT_SOURCE_DIR}/zirgen)
mlir_tablegen(zirgen/Dialect/ZHL/IR/Types.cpp.inc -gen-typedef-defs -typedefs-dialect=zhl -I${CMAKE_CURRENT_SOURCE_DIR}/zirgen)

set(LLVM_TARGET_DEFINITIONS "${CMAKE_CURRENT_SOURCE_DIR}/zirgen/zirgen/Dialect/ZHL/IR/Ops.td")
mlir_tablegen(zirgen/Dialect/ZHL/IR/Ops.h.inc -gen-op-decls -I${CMAKE_CURRENT_SOURCE_DIR}/zirgen)
mlir_tablegen(zirgen/Dialect/ZHL/IR/Ops.cpp.inc -gen-op-defs -I${CMAKE_CURRENT_SOURCE_DIR}/zirgen)

# TODO: Adapt to this repo
# llzk_add_mlir_doc(LLZKOpsDocGen Dialect.md -gen-dialect-doc -dialect=llzk)

add_public_tablegen_target(ZHLDialectIncGen)
add_dependencies(mlir-headers ZHLDialectIncGen)
add_dependencies(ZHLDialectHeaders ZHLDialectIncGen)

add_library(ZHLDialect)
file(GLOB_RECURSE ZIRGEN_CXX_FILES LIST_DIRECTORIES true "zirgen")
message(STATUS "cpp files: ${ZIRGEN_CXX_FILES}")
execute_process(COMMAND ls -la ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(ZHLDialect PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/zirgen/zirgen/Dialect/ZHL/IR/Dialect.cpp 
  ${CMAKE_CURRENT_SOURCE_DIR}/zirgen/zirgen/Dialect/ZHL/IR/Ops.cpp 
  ${CMAKE_CURRENT_SOURCE_DIR}/zirgen/zirgen/Dialect/ZHL/IR/Types.cpp
)
add_dependencies(ZHLDialect ZHLDialectHeaders)
target_link_libraries(ZHLDialect PRIVATE ZHLDialectHeaders LLVMHeaders MLIRHeaders)

############## zirgen's zir parser ###############

add_library(ZIRParser)
target_sources(ZIRParser PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/zirgen/zirgen/dsl/lexer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/zirgen/zirgen/dsl/ast.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/zirgen/zirgen/dsl/lower.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/zirgen/zirgen/dsl/parser.cpp
)
target_include_directories(ZIRParser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/zirgen)
add_dependencies(ZIRParser ZHLDialectHeaders)
target_link_libraries(ZIRParser PRIVATE ZHLDialectHeaders LLVMHeaders MLIRHeaders)

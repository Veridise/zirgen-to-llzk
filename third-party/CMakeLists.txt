# Cmake targets for the third party components obtained via the submodules in the same directory as this file.

if(NOT DEFINED ZIRGEN_SRC)
  set(ZIRGEN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/zirgen")
  message(STATUS "Using default zirgen source directory: ${ZIRGEN_SRC}")
else()
  message(STATUS "Using provided zirgen source directory: ${ZIRGEN_SRC}")
endif()

############## zirgen's ZHL Dialect ##############

# Header only library for users of the ZHL dialect
add_library(ZHLDialectHeaders INTERFACE)
target_include_directories(
  ZHLDialectHeaders SYSTEM
  INTERFACE # Headers in the ZHL IR folder
            $<BUILD_INTERFACE:${ZIRGEN_SRC}/zirgen/Dialect/ZHL/IR>
            # For the includes from the top of the repository
            $<BUILD_INTERFACE:${ZIRGEN_SRC}/>
            # TableGen'd files
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/zirgen/Dialect/ZHL/IR>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/zirgen/Dialect/ZHL/IR>)
install(
  DIRECTORY "${ZIRGEN_SRC}/zirgen/Dialect/ZHL/IR" "${CMAKE_CURRENT_BINARY_DIR}/zirgen/Dialect/ZHL/IR"
  TYPE INCLUDE
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.h.inc"
  PATTERN "*.td"
  PATTERN CMakeFiles EXCLUDE)

# TableGen's files for the ZHL dialect
include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/zirgen/Dialect/ZHL/IR")

set(LLVM_TARGET_DEFINITIONS "${ZIRGEN_SRC}/zirgen/Dialect/ZHL/IR/Dialect.td")
mlir_tablegen(zirgen/Dialect/ZHL/IR/Dialect.h.inc -gen-dialect-decls -dialect=zhl -I${ZIRGEN_SRC})
mlir_tablegen(zirgen/Dialect/ZHL/IR/Dialect.cpp.inc -gen-dialect-defs -dialect=zhl -I${ZIRGEN_SRC})

set(LLVM_TARGET_DEFINITIONS "${ZIRGEN_SRC}/zirgen/Dialect/ZHL/IR/Attrs.td")
mlir_tablegen(zirgen/Dialect/ZHL/IR/Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=zhl -I${ZIRGEN_SRC})
mlir_tablegen(zirgen/Dialect/ZHL/IR/Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=zhl -I${ZIRGEN_SRC})

set(LLVM_TARGET_DEFINITIONS "${ZIRGEN_SRC}/zirgen/Dialect/ZHL/IR/Types.td")
mlir_tablegen(zirgen/Dialect/ZHL/IR/Types.h.inc -gen-typedef-decls -typedefs-dialect=zhl -I${ZIRGEN_SRC})
mlir_tablegen(zirgen/Dialect/ZHL/IR/Types.cpp.inc -gen-typedef-defs -typedefs-dialect=zhl -I${ZIRGEN_SRC})

set(LLVM_TARGET_DEFINITIONS "${ZIRGEN_SRC}/zirgen/Dialect/ZHL/IR/Ops.td")
mlir_tablegen(zirgen/Dialect/ZHL/IR/Ops.h.inc -gen-op-decls -I${ZIRGEN_SRC})
mlir_tablegen(zirgen/Dialect/ZHL/IR/Ops.cpp.inc -gen-op-defs -I${ZIRGEN_SRC})

zklang_add_mlir_doc(ZhlOpsDocGen ZHLDialect.md -gen-dialect-doc -dialect=zhl -I${ZIRGEN_SRC})

add_public_tablegen_target(ZHLDialectIncGen)
add_dependencies(mlir-headers ZHLDialectIncGen)
add_dependencies(ZHLDialectHeaders ZHLDialectIncGen)

add_library(ZHLDialect)
target_sources(ZHLDialect PRIVATE
  ${ZIRGEN_SRC}/zirgen/Dialect/ZHL/IR/Dialect.cpp
  ${ZIRGEN_SRC}/zirgen/Dialect/ZHL/IR/Ops.cpp
  ${ZIRGEN_SRC}/zirgen/Dialect/ZHL/IR/Types.cpp
)
add_dependencies(ZHLDialect ZHLDialectHeaders)
target_link_libraries(ZHLDialect PRIVATE ZHLDialectHeaders LLVMHeaders MLIRHeaders)

############## zirgen's zir parser ###############

add_library(ZIRParser)
target_sources(ZIRParser PRIVATE
  ${ZIRGEN_SRC}/zirgen/dsl/lexer.cpp
  ${ZIRGEN_SRC}/zirgen/dsl/ast.cpp
  ${ZIRGEN_SRC}/zirgen/dsl/lower.cpp
  ${ZIRGEN_SRC}/zirgen/dsl/parser.cpp
)
target_include_directories(ZIRParser PUBLIC ${ZIRGEN_SRC})
add_dependencies(ZIRParser ZHLDialectHeaders)
target_link_libraries(ZIRParser PRIVATE ZHLDialectHeaders LLVMHeaders MLIRHeaders)

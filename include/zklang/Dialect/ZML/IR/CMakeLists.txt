include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})


set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=zml)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=zml)

set(LLVM_TARGET_DEFINITIONS "Attrs.td")
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=zml)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=zml)

set(LLVM_TARGET_DEFINITIONS "Types.td")
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=zml)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=zml)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS "OpInterfaces.td")
mlir_tablegen(OpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(ZMLDialectIncGen)
add_dependencies(mlir-headers ZMLDialectIncGen)
add_dependencies(ZklangHeaders ZMLDialectIncGen)

add_library(ZIRC)

target_link_libraries(ZIRC PRIVATE LLVMHeaders MLIRHeaders ZHLDialectHeaders ZHLDialect ZklangHeaders)
target_include_directories(ZIRC PUBLIC ${ZKLANG_INCLUDE_DIR})
add_dependencies(ZIRC ZHLDialect ZklangHeaders)

add_subdirectory(Dialect)
add_subdirectory(Passes)
add_subdirectory(Frontends)

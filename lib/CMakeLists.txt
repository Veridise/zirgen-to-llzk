add_library(ZIRC)

target_link_libraries(ZIRC PRIVATE LLVMHeaders MLIRHeaders ZHLDialectHeaders ZHLDialect ZIRCHeaders)
target_include_directories(ZIRC PUBLIC ${ZIRC_INCLUDE_DIR})
add_dependencies(ZIRC ZHLDialect ZIRCHeaders)

add_subdirectory(Dialect)

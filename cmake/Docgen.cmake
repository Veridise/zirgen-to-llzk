macro(zklang_setup_doc_generation DOXYGEN_DEPS)

set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc")
set(ZKLANG_MLIR_DOC_OUTPUT_DIR "${DOXYGEN_OUTPUT_DIRECTORY}/mlir")

# Documentation
add_custom_target(doc)
find_package(Doxygen OPTIONAL_COMPONENTS dot)
if(Doxygen_FOUND)
  message(STATUS "Doxygen found, enabling documentation...")
  set(DOXYGEN_EXTRACT_ALL YES)
  set(DOXYGEN_INCLUDE_PATH "${CMAKE_CURRENT_BINARY_DIR}/include/")
  set(DOXYGEN_EXCLUDE_PATTERNS ${CMAKE_CURRENT_BINARY_DIR}/include/*/*.md)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE
      ${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/index.md)
  set(DOXYGEN_FILE_PATTERNS
      *.cpp
      *.cpp.inc
      *.h.inc
      *.hpp
      *.h
      *.td
      *.md
      *.py)
  set(DOXYGEN_EXTENSION_MAPPING inc=C++)
  set(DOXYGEN_MACRO_EXPANSION YES)
  set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
  set(DOXYGEN_PREDEFINED GET_OP_CLASSES GET_TYPEDEF_CLASSES GET_ATTR_CLASSES)
  set(DOXYGEN_SOURCE_BROWSER YES)
  set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
  doxygen_add_docs(
    doxygen
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/"
    "${CMAKE_CURRENT_BINARY_DIR}/include/"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    "${ZKLANG_MLIR_DOC_OUTPUT_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/")
  add_dependencies(doxygen ${DOXYGEN_DEPS})
  add_dependencies(doc doxygen)
endif()

endmacro()

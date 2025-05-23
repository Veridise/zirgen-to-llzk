set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc")
set(ZKLANG_DOXYGEN_BUILD_OUTPUT "${DOXYGEN_OUTPUT_DIRECTORY}/doxygen")
set(ZKLANG_MLIR_DOC_OUTPUT_DIR "${ZKLANG_DOXYGEN_BUILD_OUTPUT}/mlir")

macro(zklang_setup_doc_generation DOXYGEN_DEPS)
  find_package(Doxygen OPTIONAL_COMPONENTS dot)
  if(Doxygen_FOUND)
    message(STATUS "Doxygen found, enabling documentation...")
    add_custom_target(doc)

    # Fetch style document
    include(FetchContent)
    set(DoxygenAwesomeCSS_SOURCE_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/deps/doxygen-awesome-css")
    FetchContent_Declare(
      DoxygenAwesomeCSS
      GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
      GIT_TAG v2.3.4
      SOURCE_DIR "${DoxygenAwesomeCSS_SOURCE_DIR}")
    FetchContent_MakeAvailable(DoxygenAwesomeCSS)

    # * doxygen awesome settings
    set(DOXYGEN_HTML_EXTRA_STYLESHEET
        "${DoxygenAwesomeCSS_SOURCE_DIR}/doxygen-awesome.css"
        "${DoxygenAwesomeCSS_SOURCE_DIR}/doxygen-awesome-sidebar-only.css"
        "${DoxygenAwesomeCSS_SOURCE_DIR}/doxygen-awesome-sidebar-only-darkmode-toggle.css"
    )
    set(DOXYGEN_HTML_EXTRA_FILES
        "${DoxygenAwesomeCSS_SOURCE_DIR}/doxygen-awesome-darkmode-toggle.js"
        "${DoxygenAwesomeCSS_SOURCE_DIR}/doxygen-awesome-paragraph-link.js")
    set(DOXYGEN_HTML_HEADER
        "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/header.html")
    set(DOXYGEN_HTML_FOOTER
        "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/footer.html")
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_DISABLE_INDEX NO)
    set(DOXYGEN_FULL_SIDEBAR NO)
    set(DOXYGEN_HTML_COLORSTYLE "LIGHT")
    set(DOXYGEN_DOT_IMAGE_FORMAT "svg")
    set(DOXYGEN_DOT_TRANSPARENT YES)
    set(DOXYGEN_TREEVIEW_WIDTH 700)
    set(DOXYGEN_FULL_PATH_NAMES YES)

    # Remaining doxygen setup
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_INCLUDE_PATH "${CMAKE_CURRENT_BINARY_DIR}/include/")
    set(DOXYGEN_EXCLUDE_PATTERNS
        "${CMAKE_CURRENT_BINARY_DIR}/include/*/*.md"
        # We ignore the passes because we aggregate the documentation under
        # `zklang-opt`
        "${ZKLANG_MLIR_DOC_OUTPUT_DIR}/passes/*.md"
        # Same for Dialect docs
        "${ZKLANG_MLIR_DOC_OUTPUT_DIR}/*Dialect.md"
    )

    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE
        "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/index.md")
    set(DOXYGEN_FILE_PATTERNS
        *.cpp
        *.cpp.inc
        *.h.inc
        *.hpp
        *.h
        *.td
        *.md
        *.py
        *.txt)
    set(DOXYGEN_EXTENSION_MAPPING inc=C++)
    set(DOXYGEN_MACRO_EXPANSION YES)
    set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
    set(DOXYGEN_PREDEFINED GET_OP_CLASSES GET_TYPEDEF_CLASSES GET_ATTR_CLASSES)
    set(DOXYGEN_SOURCE_BROWSER YES)
    set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
    doxygen_add_docs(
      doxygen
      "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen"
      "${CMAKE_CURRENT_SOURCE_DIR}/lib/"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/"
      "${CMAKE_CURRENT_BINARY_DIR}/include/"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/"
      "${ZKLANG_MLIR_DOC_OUTPUT_DIR}"
      "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

    add_dependencies(doxygen ${DOXYGEN_DEPS})
    add_dependencies(doc doxygen)
  endif()
endmacro()

function(zklang_add_mlir_doc target_name out_filename tblgen_flags)
  # this is a modified version of add_mlir_doc from AddMLIR.cmake
  set(OUT_FILE "${ZKLANG_MLIR_DOC_OUTPUT_DIR}/${out_filename}")
  tablegen(MLIR ${out_filename} ${tblgen_flags} ${ARGN})
  add_custom_command(
    OUTPUT ${OUT_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_BINARY_DIR}/${out_filename}" "${OUT_FILE}"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${out_filename}")
  add_custom_target(${target_name} DEPENDS ${OUT_FILE})
  add_dependencies(mlir-doc ${target_name})
endfunction()

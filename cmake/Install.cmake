macro(zklang_setup_install_targets)
  export(
    EXPORT ${ZKLANG_EXPORT_TARGETS}
    FILE ${CMAKE_CURRENT_BINARY_DIR}/${ZKLANG_EXPORT_TARGETS}.cmake
    NAMESPACE ZKLANG::)
  install(
    EXPORT ${ZKLANG_EXPORT_TARGETS}
    NAMESPACE ZKLANG::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/zklang)
endmacro()

# Inspired by https://github.com/cpp-best-practices/cmake_template/blob/main/cmake/CompilerWarnings.cmake 

function(
  zklang_set_project_warnings
  project_name
  WARNINGS_AS_ERRORS
  CLANG_WARNINGS
  GCC_WARNINGS)

  if("${CLANG_WARNINGS}" STREQUAL "")
    set(CLANG_WARNINGS
        -Wall
        -Wextra # reasonable and standard
        -Wshadow # warn the user if a variable declaration shadows one from a parent context
        -Wnon-virtual-dtor # warn the user if a class with virtual functions has a non-virtual destructor. This helps
        # catch hard to track down memory errors
        -Wold-style-cast # warn for c-style casts
        -Wcast-align # warn for potential performance problem casts
        -Wunused # warn on anything being unused
        -Woverloaded-virtual # warn if you overload (not override) a virtual function
        -Wpedantic # warn if non-standard C++ is used
        -Wconversion # warn on type conversions that may lose data
        -Wsign-conversion # warn on sign conversions
        -Wnull-dereference # warn if a null dereference is detected
        -Wdouble-promotion # warn if float is implicit promoted to double
        -Wformat=2 # warn on security issues around functions that format output (ie printf)
        -Wimplicit-fallthrough # warn on statements that fallthrough without an explicit annotation
        -Wswitch
        -Wreturn-type
    )
  endif()

  if("${GCC_WARNINGS}" STREQUAL "")
    set(GCC_WARNINGS
        ${CLANG_WARNINGS}
        -Wmisleading-indentation # warn if indentation implies blocks where blocks do not exist
        -Wduplicated-cond # warn if if / else chain has duplicated conditions
        -Wduplicated-branches # warn if if / else branches have duplicated code
        -Wlogical-op # warn about logical operations being used where bitwise were probably wanted
        -Wuseless-cast # warn if you perform a cast to the same type
        -Wsuggest-override # warn if an overridden member function is not marked 'override' or 'final'
    )
  endif()

  if(WARNINGS_AS_ERRORS)
    message(TRACE "Warnings are treated as errors")
    list(APPEND CLANG_WARNINGS -Werror)
    # Even though all MLIR/LLVM includes are marked with SYSTEM, the GCC compiler still produces a
    # few `null-dereference` warnings that originate in MLIR code so don't convert those to error.
    list(APPEND GCC_WARNINGS -Werror -Wno-error=null-dereference)
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(PROJECT_WARNINGS_CXX ${CLANG_WARNINGS})
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(PROJECT_WARNINGS_CXX ${GCC_WARNINGS})
  else()
    message(AUTHOR_WARNING "No compiler warnings set for CXX compiler: '${CMAKE_CXX_COMPILER_ID}'")
  endif()

  # use the same warning flags for C
  set(PROJECT_WARNINGS_C "${PROJECT_WARNINGS_CXX}")

  target_compile_options(
    ${project_name}
    INTERFACE # C++ warnings
              $<$<COMPILE_LANGUAGE:CXX>:${PROJECT_WARNINGS_CXX}>
              # C warnings
              $<$<COMPILE_LANGUAGE:C>:${PROJECT_WARNINGS_C}>)
endfunction()

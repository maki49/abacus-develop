
if(ENABLE_LCAO)
  if(BUILD_TESTING)
    add_subdirectory(test)
  endif()
    list(APPEND objects
      density_matrix.cpp
    )
  if(ENABLE_COVERAGE)
    add_coverage(lr_utils)
  endif()
endif()






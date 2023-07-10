
if(ENABLE_LCAO)
  if(BUILD_TESTING)
    add_subdirectory(test)
  endif()
  if(ENABLE_COVERAGE)
    add_coverage(lr_utils)
  endif()
endif()






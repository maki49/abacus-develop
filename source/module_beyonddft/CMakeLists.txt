add_subdirectory(AX)
add_subdirectory(dm_trans)
add_subdirectory(utils)
if(ENABLE_LCAO)
  list(APPEND objects
      hsolver_lrtd.cpp
  )
endif()
add_library(
    lr
    OBJECT
    ${objects}
)
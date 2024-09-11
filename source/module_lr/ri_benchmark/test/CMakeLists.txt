if (ENABLE_LIBRI)
  AddTest(
    TARGET ri_benchmark_test
    LIBS psi base ${math_libs} device container
    SOURCES ri_benchmark_test.cpp
  )
endif()
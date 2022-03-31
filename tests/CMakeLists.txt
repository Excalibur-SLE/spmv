function(add_spmv_test)
  # Define the supported set of keywords
  set(prefix ARG)
  set(noValues ENABLE_OPENMP ENABLE_HIPSYCL ENABLE_DPCPP)
  set(singleValues TARGET)
  set(multiValues SOURCES)

  # Process the arguments passed in
  include(CMakeParseArguments)
  cmake_parse_arguments(${prefix}
    "${noValues}"
    "${singleValues}"
    "${multiValues}"
    ${ARGN})

  add_executable(${ARG_TARGET} ${ARG_SOURCES})
  if(ARG_ENABLE_HIPSYCL)
    add_sycl_to_target(TARGET ${ARG_TARGET})
  endif()
  target_include_directories(${ARG_TARGET} PRIVATE ${CMAKE_BINARY_DIR}/spmv)
  target_link_libraries(${ARG_TARGET} PRIVATE ${PROJECT_NAME}::${LIB_NAME})
  foreach(numProcs RANGE 1 2)
    add_test(NAME ${ARG_TARGET}_n${numProcs} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${numProcs} ${MPIEXEC_PREFLAGS} ./${ARG_TARGET})
    if(ARG_ENABLE_HIPSYCL OR ARG_ENABLE_OPENMP)
      set_tests_properties(${ARG_TARGET}_n${numProcs} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
    endif()
    if(ARG_ENABLE_DPCPP)
      set_tests_properties(${ARG_TARGET}_n${numProcs} PROPERTIES ENVIRONMENT "DPCPP_CPU_NUM_CUS=2")
    endif()
  endforeach()
endfunction()

# Define tests
add_spmv_test(TARGET test_spmv SOURCES test_spmv.cpp)
if(ENABLE_OPENMP)
  add_spmv_test(TARGET test_spmv_openmp SOURCES test_spmv_openmp.cpp ENABLE_OPENMP)
endif()
if(ENABLE_OPENMP_OFFLOAD)
  add_spmv_test(TARGET test_spmv_openmp_offload SOURCES test_spmv_openmp_offload.cpp)
elseif(ENABLE_DPCPP)
  add_spmv_test(TARGET test_spmv_sycl SOURCES test_spmv_sycl.cpp ENABLE_DPCPP)
elseif(ENABLE_HIPSYCL)
  add_spmv_test(TARGET test_spmv_sycl SOURCES test_spmv_sycl.cpp ENABLE_HIPSYCL)
elseif(ENABLE_CUDA)
  add_spmv_test(TARGET test_spmv_cuda SOURCES test_spmv_cuda.cpp)
endif()

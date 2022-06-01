function(add_demo)
  # Define the supported set of keywords
  set(prefix ARG)
  set(noValues ENABLE_HIPSYCL ENABLE_DPCPP)
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
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    # Can use imported target BLAS::BLAS for version >= 3.18
    target_link_libraries(${ARG_TARGET} PRIVATE BLAS::BLAS)
  else()
    target_link_libraries(${ARG_TARGET} PRIVATE ${BLAS_LIBRARIES})
  endif()
  if(ARG_ENABLE_HIPSYCL)
    add_sycl_to_target(TARGET ${ARG_TARGET})
  endif()
  if(ARG_ENABLE_DPCPP)
    target_link_libraries(${ARG_TARGET} PRIVATE MKL::MKL_DPCPP)
  endif()
  target_link_libraries(${ARG_TARGET} PRIVATE ${PROJECT_NAME}::${LIB_NAME})
  install(TARGETS ${ARG_TARGET} DESTINATION ${CMAKE_INSTALL_BINDIR}/demos)
endfunction()

# Add demos
add_demo(TARGET demo_spmv SOURCES spmv.cpp)
add_demo(TARGET demo_cg SOURCES cg.cpp)
if(ENABLE_OPENMP)
  add_demo(TARGET demo_spmv_openmp SOURCES spmv_openmp.cpp)
  add_demo(TARGET demo_cg_openmp SOURCES cg_openmp.cpp)
endif()
if(ENABLE_OPENACC)
  add_demo(TARGET demo_spmv_openacc SOURCES spmv_openacc.cpp)
endif()
if(ENABLE_DPCPP)
  add_demo(TARGET demo_spmv_sycl SOURCES spmv_sycl.cpp ENABLE_DPCPP)
  add_demo(TARGET demo_cg_sycl SOURCES cg_sycl.cpp ENABLE_DPCPP)
endif()
if(ENABLE_HIPSYCL)
  add_demo(TARGET demo_spmv_sycl SOURCES spmv_sycl.cpp ENABLE_HIPSYCL)
  add_demo(TARGET demo_cg_sycl SOURCES cg_sycl.cpp ENABLE_HIPSYCL)
endif()
if(ENABLE_CUDA)
  add_demo(TARGET demo_spmv_cuda SOURCES spmv_cuda.cpp)
  add_demo(TARGET demo_cg_cuda SOURCES cg_cuda.cpp)
endif()

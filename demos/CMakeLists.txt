# Executables
add_executable(spmv_demo spmv.cpp CreateA.cpp)
target_link_libraries(spmv_demo PRIVATE ${PROJECT_NAME}::${LIB_NAME})
install(TARGETS spmv_demo DESTINATION ${CMAKE_INSTALL_BINDIR}/demos)

add_executable(cg_demo cg.cpp)
set_target_properties(cg_demo PROPERTIES
  CXX_STANDARD 17
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
  )
target_link_libraries(cg_demo PRIVATE ${PROJECT_NAME}::${LIB_NAME})
install(TARGETS cg_demo DESTINATION ${CMAKE_INSTALL_BINDIR}/demos)

add_executable(restrictmain restrictmain.cpp)
target_link_libraries(restrictmain PRIVATE ${PROJECT_NAME}::${LIB_NAME})

if(ENABLE_DPCPP OR ENABLE_HIPSYCL)
  add_executable(spmv_demo_sycl spmv_sycl.cpp CreateA.cpp)
  target_link_libraries(spmv_demo_sycl PRIVATE ${PROJECT_NAME}::${LIB_NAME})
  install(TARGETS spmv_demo_sycl DESTINATION ${CMAKE_INSTALL_BINDIR}/demos)

  # add_executable(cg_demo_sycl cg_sycl.cpp)
  # target_link_libraries(cg_demo_sycl PRIVATE ${PROJECT_NAME}::${LIB_NAME})
  # install(TARGETS cg_demo_sycl DESTINATION ${CMAKE_INSTALL_BINDIR}/demos)

  if(ENABLE_HIPSYCL)
    add_sycl_to_target(TARGET spmv_demo_sycl SOURCES spmv_sycl.cpp)
    #add_sycl_to_target(TARGET cg_demo_sycl SOURCES cg_sycl.cpp)
  endif()
elseif(ENABLE_OPENMP)
  add_executable(spmv_demo_openmp spmv_openmp.cpp CreateA.cpp)
  target_link_libraries(spmv_demo_openmp PRIVATE ${PROJECT_NAME}::${LIB_NAME})
  install(TARGETS spmv_demo_openmp DESTINATION ${CMAKE_INSTALL_BINDIR}/demos)

  add_executable(cg_demo_openmp cg_openmp.cpp)
  set_target_properties(cg_demo_openmp PROPERTIES
    CXX_STANDARD 17
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    )
  target_link_libraries(cg_demo_openmp PRIVATE ${PROJECT_NAME}::${LIB_NAME})
  install(TARGETS cg_demo_openmp DESTINATION ${CMAKE_INSTALL_BINDIR}/demos)
elseif(ENABLE_OPENACC)
  add_executable(spmv_demo_openacc spmv_openacc.cpp)
  target_link_libraries(spmv_demo_openacc PRIVATE ${PROJECT_NAME}::${LIB_NAME})
elseif(ENABLE_CUDA)
  add_executable(spmv_demo_cuda spmv_cuda.cpp CreateA.cpp)
  target_link_libraries(spmv_demo_cuda PRIVATE ${PROJECT_NAME}::${LIB_NAME})
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    target_link_libraries(spmv_demo_cuda PRIVATE CUDA::cublas)
  else()
    target_link_libraries(spmv_demo_cuda PRIVATE ${CUBLAS_LIB})
  endif()
  install(TARGETS spmv_demo_cuda DESTINATION ${CMAKE_INSTALL_BINDIR}/demos)

  add_executable(cg_demo_cuda cg_cuda.cpp)
  set_target_properties(cg_demo_cuda PROPERTIES
    CXX_STANDARD 17
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    )
  target_link_libraries(cg_demo_cuda PRIVATE ${PROJECT_NAME}::${LIB_NAME})
  install(TARGETS cg_demo_cuda DESTINATION ${CMAKE_INSTALL_BINDIR}/demos)
endif()

add_library(${PROJECT_NAME} SHARED
  ${PROJECT_SOURCE_DIR}/spmv/L2GMap.cpp
  ${PROJECT_SOURCE_DIR}/spmv/Matrix.cpp
  ${PROJECT_SOURCE_DIR}/spmv/read_petsc.cpp
  ${PROJECT_SOURCE_DIR}/spmv/cg.cpp
  )

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "spmv.h")
set_target_properties(${PROJECT_NAME} PROPERTIES PRIVATE_HEADER "L2GMap.h;Matrix.h;read_petsc.h;cg.h")

if (USE_DPCPP OR USE_HIPSYCL)
  add_subdirectory(sycl)
elseif (USE_OPENMP_OFFLOAD)
  add_subdirectory(openmp_offload)
elseif (USE_CUDA)
  add_subdirectory(cuda)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} INTERFACE $<INSTALL_INTERFACE:include>)

target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen MPI::MPI_C)
if (OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
if (BLAS_FOUND)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_MKL)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${BLAS_LIBRARIES})
endif()

install(TARGETS ${PROJECT_NAME}
  EXPORT libspmv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include
  PRIVATE_HEADER DESTINATION include
  )

# First attempt at installing a config
include(CMakePackageConfigHelpers)
write_basic_package_version_file( 
  spmvConfig.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion
  )

install(EXPORT libspmv
  FILE spmvConfig.cmake
  NAMESPACE spmv::
  DESTINATION share/spmv/cmake)

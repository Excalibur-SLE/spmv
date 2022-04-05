# Configure the library's main header file
if(ENABLE_OPENMP)
  set(_OPENMP_HOST 1)
endif()
if(ENABLE_OPENMP_OFFLOAD)
  set(_OPENMP_OFFLOAD 1)
endif()
if(ENABLE_HIPSYCL OR ENABLE_DPCPP)
  set(_SYCL 1)
endif()
if(ENABLE_CUDA)
  set(_CUDA 1)
endif()
configure_file(${PROJECT_SOURCE_DIR}/spmv/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

add_library(${LIB_NAME} SHARED
  ${PROJECT_SOURCE_DIR}/spmv/device_executor.cpp
  ${PROJECT_SOURCE_DIR}/spmv/reference_executor.cpp
  ${PROJECT_SOURCE_DIR}/spmv/coo_matrix.cpp
  ${PROJECT_SOURCE_DIR}/spmv/coo_kernels.cpp
  ${PROJECT_SOURCE_DIR}/spmv/csr_matrix.cpp
  ${PROJECT_SOURCE_DIR}/spmv/csr_kernels.cpp
  ${PROJECT_SOURCE_DIR}/spmv/L2GMap.cpp
  ${PROJECT_SOURCE_DIR}/spmv/Matrix.cpp
  ${PROJECT_SOURCE_DIR}/spmv/read_petsc.cpp
  ${PROJECT_SOURCE_DIR}/spmv/cg.cpp
  )

# Create an alias to be used within the project and by other projects.
# Namespaced names must be CMake targets, so wrong use of the library
# name will be triggered at generation time instead of link time.
# It is therefore more robust to link to namespaced names.
add_library(${PROJECT_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})

# Set header file search path for build and install directories
# This is not needed because the header files are in the same directory
# as the source files
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

# Create an export header to control symbol visibility.
include(GenerateExportHeader)
generate_export_header(${LIB_NAME})

# Set the header files that need to be installed
set_target_properties(${LIB_NAME}
  PROPERTIES PUBLIC_HEADER
    "spmv.h;L2GMap.h;Matrix.h;read_petsc.h;cg.h;device_executor.h;reference_executor.h"
  )

# Set optional definitions
target_compile_definitions(${LIB_NAME}
  PRIVATE
    $<$<BOOL:${ENABLE_OPENMP}>:_OPENMP_HOST>
    $<$<BOOL:${ENABLE_OPENMP_OFFLOAD}>:_OPENMP_OFFLOAD>
    $<$<OR:$<BOOL:${ENABLE_HIPSYCL}>,$<BOOL:${ENABLE_DPCPP}>>:_SYCL>
    $<$<BOOL:${ENABLE_CUDA}>:_CUDA>
    $<$<BOOL:${BLAS_FOUND}>:_MKL>
  )
# Eigen-specific definitions
target_compile_definitions(${LIB_NAME}
  PUBLIC "EIGEN_MAX_ALIGN_BYTES=32"
  )

# Set library dependencies
target_link_libraries(${LIB_NAME}
  PUBLIC Eigen3::Eigen MPI::MPI_C MPI::MPI_CXX
  PRIVATE $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
  )
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  # Can use imported target BLAS::BLAS for version >= 3.18
  target_link_libraries(${LIB_NAME}
    PRIVATE $<$<BOOL:${BLAS_FOUND}>:BLAS::BLAS>
    )
else()
  target_link_libraries(${LIB_NAME}
    PRIVATE $<$<BOOL:${BLAS_FOUND}>:${BLAS_LIBRARIES}>
    )
endif()

# Include optional code
if(ENABLE_DPCPP OR ENABLE_HIPSYCL)
  add_subdirectory(sycl)
elseif(ENABLE_OPENMP)
  add_subdirectory(openmp)
elseif(ENABLE_OPENMP_OFFLOAD)
  add_subdirectory(openmp_offload)
elseif(ENABLE_CUDA)
  add_subdirectory(cuda)
endif()

install(TARGETS ${LIB_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}
  )

install(EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Config.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
  )

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}_export.h
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}
  )

target_sources(${LIB_NAME} PRIVATE omp_executor.cpp csr_kernels.openmp.cpp)

# Extend header path for build and install phases
target_include_directories(${LIB_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )
target_include_directories(${LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}/openmp>
  )

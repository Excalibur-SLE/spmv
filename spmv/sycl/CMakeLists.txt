add_sycl_to_target(TARGET ${LIB_NAME})
target_sources(${LIB_NAME} PRIVATE sycl_executor.cpp csr_kernels.sycl.cpp)

# Extend header path for build and install phases
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}/sycl>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

# Install the header files manually, since CMake does not respect subdirectories
set(publicHeaders "sycl_executor.h;cg_sycl.h")
foreach(header ${publicHeaders})
  install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}/sycl")
endforeach()

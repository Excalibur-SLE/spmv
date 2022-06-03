target_sources(${LIB_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_executor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_executor.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/coo_kernels.cuda.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csr_kernels.cuda.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cg.cuda.cu
  )

# Extend header path for build and install phases
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}/cuda>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

# We are forced to disable pedantic with NVCC as it generates non-suppressible warnings of type
# "warning: style of line directive is a GCC extension"
# There is no workaround for -Wpedantic with gcc/nvcc and no immediate plans to fix this
target_compile_options(${LIB_NAME}
  PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Wno-pedantic -Wno-unused-parameter>
  )

# The CMAKE_CUDA_ARCHITECTURES variable is initialized automatically when
# CMAKE_CUDA_COMPILER_ID is NVIDIA for CMake >= 3.18
if(CMAKE_VERSION VERSION_LESS 3.18)
  foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    target_compile_options(${LIB_NAME}
      PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_${arch},code=sm_${arch}>
      )
    target_link_options(${LIB_NAME}
      PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_${arch},code=sm_${arch}>
      )
  endforeach()
endif()

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC" OR CMAKE_CXX_COMPILER_ID MATCHES "PGI")
  target_link_libraries(${LIB_NAME}
    PUBLIC CUDA::cudart ${CUBLAS_LIB} ${CUSPARSE_LIB}
    )
else()
  target_link_libraries(${LIB_NAME}
    PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse
    )
  target_include_directories(${LIB_NAME} PUBLIC "${CUDAToolkit_ROOT}/../math_libs/include")
endif()

# Install the header files manually, since CMake does not respect subdirectories
set(publicHeaders "cg_cuda.h")
foreach(header ${publicHeaders})
  install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}/cuda")
endforeach()

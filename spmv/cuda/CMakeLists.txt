target_sources(${LIB_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_executor.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/coo_kernels.cuda.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csr_kernels.cuda.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cg.cu
  )

# Extend header path for build and install phases
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}/cuda>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

# Set the header files that need to be installed (cannot use property
# in this cuda as it does not preserve directory structure)
set(publicHeaders "cg_cuda.h")
foreach(header ${publicHeaders})
  install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}/cuda")
endforeach()

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
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
  target_link_libraries(${LIB_NAME}
    PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse
    )
else()
  target_link_libraries(${LIB_NAME}
    PUBLIC ${CUDART_LIB}
    PRIVATE ${CUBLAS_LIB} ${CUSPARSE_LIB}
    )
endif()

target_sources(${PROJECT_NAME} PRIVATE
  L2GMap.cu
  Matrix.cu
  cg.cu
  )

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# FIXME not working
set_target_properties(${PROJECT_NAME} PROPERTIES PRIVATE_HEADER "cuda/helper_cuda.h;cuda/cg_cuda.h")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# We are forced to disable pedantic with NVCC as it generates non-suppressible warnings of type
# "warning: style of line directive is a GCC extension"
# There is no workaround for -Wpedantic with gcc/nvcc and no immediate plans to fix this
target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
  -Wno-pedantic
  --ptxas-options=-v
  --generate-line-info
  --relocatable-device-code=true
  >)

target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas CUDA::cusparse)
# Workaround for CUDA::cuBLAS CUDA::cuSPARSE for NVHPC
# target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
# target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDAToolkit_ROOT}/../math_libs/lib64/libcublas.so)
# target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDAToolkit_ROOT}/../math_libs/lib64/libcusparse.so)
#target_include_directories(${PROJECT_NAME} PUBLIC ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_ROOT}/../math_libs/include)

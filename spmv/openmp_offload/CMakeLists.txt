target_sources(${LIB_NAME} PRIVATE cg.openmp.cpp)

# Extend header path for build and install phases
target_include_directories(${LIB_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )
target_include_directories(${LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/openmp_offload>
  )

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  # Before GCC-12 use undocumented -foffload=nvptx-none=-latomic, otherwise -foffload-options=nvptx-none=-latomic
  target_compile_options(${LIB_NAME} PUBLIC -fopenmp -fno-stack-protector -foffload=nvptx-none -foffload=nvptx-none=-latomic)
  target_link_options(${LIB_NAME} PUBLIC -fopenmp -fno-stack-protector -foffload=nvptx-none -foffload=nvptx-none=-latomic)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # See related bug: https://gitlab.kitware.com/cmake/cmake/-/issues/21787
  # libomp needs to be in linker's search path, when Clang is built in non-standard location
  target_compile_options(${LIB_NAME} PUBLIC -fopenmp=libomp -fopenmp-targets=nvptx64 -Wno-unknown-cuda-version --cuda-path=${CUDAToolkit_ROOT})
  target_link_options(${LIB_NAME} PUBLIC -fopenmp=libomp -fopenmp-targets=nvptx64 -Wno-unknown-cuda-version --cuda-path=${CUDAToolkit_ROOT})
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "PGI")
  target_compile_options(${LIB_NAME} PUBLIC -mp=gpu -Minfo=mp)
endif()

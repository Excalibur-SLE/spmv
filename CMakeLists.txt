# Require CMake 3.17
cmake_minimum_required(VERSION 3.17)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME spmv)
set(PACKAGE_VERSION 1.0)
project(${PROJECT_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES CXX C)

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)

#set(CMAKE_SKIP_RPATH FALSE)
#set(CMAKE_SKIP_BUILD_RPATH FALSE)
#set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#set(CMAKE_INSTALL_RPATH "")
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(MPI REQUIRED COMPONENTS C)

option(USE_OPENMP "Use OpenMP for shared-memory parallelization" OFF)
option(USE_OPENMP_OFFLOAD "Use OpenMP for accelerator offloading" OFF)
option(USE_DPCPP "Use DPC++ for shared-memory parallelization" OFF)
option(USE_HIPSYCL "Use hipSYCL for shared-memory parallelization" OFF)
option(USE_MKL_SEQUENTIAL "Use sequential (non-threaded) Intel MKL to accelerate SpMV kernels whenever possible" OFF)
option(USE_MKL_PARALLEL "Use threaded Intel MKL to accelerate SpMV kernels whenever possible" OFF)
option(USE_CUDA "Use CUDA (cuBLAS, cuSPARSE) for GPU offloading" OFF)

if (USE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if (OpenMP_CXX_FOUND)
    if ("${OpenMP_CXX_LIBRARIES}" STREQUAL "")
      message(STATUS "No OpenMP library needs to be linked against")
    else()
      message(STATUS "Will link against OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")
    endif()
    add_compile_definitions(_OPENMP_HOST)
  else()
    message(WARNING "OpenMP was requested, but not found.")
  endif()
endif()

if (USE_OPENMP_OFFLOAD)
  find_package(OpenMP COMPONENTS CXX)
  if (OpenMP_CXX_FOUND)
    if ("${OpenMP_CXX_LIBRARIES}" STREQUAL "")
      message(STATUS "No OpenMP library needs to be linked against")
    else()
      message(STATUS "Will link against OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")
    endif()
    add_compile_definitions(_OPENMP_OFFLOAD)
  else()
    message(WARNING "OpenMP was requested, but not found.")
  endif()
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # Before GCC-12 use undocumented -foffload=nvptx-none=-latomic, otherwise -foffload-options=nvptx-none=-latomic
    add_compile_options(-fopenmp -fno-stack-protector -foffload=nvptx-none -foffload=nvptx-none=-latomic)
    add_link_options(-fopenmp -fno-stack-protector -foffload=nvptx-none -foffload=nvptx-none=-latomic)
  endif()
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # See related bug: https://gitlab.kitware.com/cmake/cmake/-/issues/21787
    # libomp needs to be in linker's search path, when Clang is built in non-standard location
    add_compile_options(-fopenmp=libomp -fopenmp-targets=nvptx64 -Wno-unknown-cuda-version --cuda-path=${CUDAToolkit_ROOT})
    add_link_options(-fopenmp=libomp -fopenmp-targets=nvptx64 -Wno-unknown-cuda-version --cuda-path=${CUDAToolkit_ROOT})
  endif()
  if (CMAKE_CXX_COMPILER_ID MATCHES "PGI")
    add_compile_options(-mp=gpu -Minfo=mp)
  endif()
endif()

if (USE_DPCPP)
  if (DEFINED ENV{ONEAPI_ROOT})
    message(STATUS "Using DPC++ toolchain in: $ENV{ONEAPI_ROOT}")
  else()
    message(FATAL_ERROR "Please set ONEAPI_ROOT using the vendor-provided script to enable use of the DPC++ toolchain.")
  endif()
  # Check compiler is set to dpcpp
  if (NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(FATAL_ERROR "Please set dpcpp as the C++ compiler.")
  endif()
  add_compile_definitions(_SYCL)
  add_compile_definitions(_DPCPP)
  add_compile_options(-fsycl -sycl-std=2020 -fsycl-unnamed-lambda)
  # Enable AOT compilation
  # x86_64 CPUs
  #add_compile_options(-fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice)
  #add_link_options(-fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice)
  # Intel GPUs
  #add_compile_options(-fsycl-targets=spir64_gen-unknown-unknown-sycldevice)
  #add_link_options(-fsycl-targets=spir64_gen-unknown-unknown-sycldevice)
  # NVIDIA GPUs
  #find_package(CUDAToolkit)
  #add_compile_options(-fsycl-targets=nvptx64-nvidia-cuda --cuda-path=${CUDAToolkit_ROOT})
  #add_link_options(-fsycl-targets=nvptx64-nvidia-cuda --cuda-path=${CUDAToolkit_ROOT})
endif()

if (USE_HIPSYCL)
  find_package(hipSYCL CONFIG REQUIRED)
  find_package(CUDAToolkit)
  # We rely on the __HIPSYCL__ macro
  add_compile_definitions(_SYCL)
  add_compile_options(-Wno-unknown-cuda-version -Wno-unknown-attributes)
endif()

# Can use imported target BLAS::BLAS for version >= 3.18
if (USE_MKL_SEQUENTIAL)
  set(BLA_VENDOR Intel10_64lp_seq)
  find_package(BLAS)
  if (NOT BLAS_FOUND)
    message(WARNING "Intel MKL was requested, but not found.")
  endif()
endif()

# Can use imported target BLAS::BLAS for version >= 3.18
if (USE_MKL_PARALLEL)
  set(BLA_VENDOR Intel10_64lp)
  find_package(BLAS)
  if (NOT BLAS_FOUND)
    message(WARNING "Intel MKL was requested, but not found.")
  endif()
endif()

if (USE_CUDA)
  enable_language(CUDA)
  # FIXME: Requires CMake version >= 3.17
  find_package(CUDAToolkit REQUIRED)
  add_compile_definitions(USE_CUDA)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_options(-DEIGEN_MAX_ALIGN_BYTES=32)

add_subdirectory(spmv)
include_directories(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(demos)
enable_testing()
add_subdirectory(tests)

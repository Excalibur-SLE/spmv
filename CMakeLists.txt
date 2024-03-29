# Require CMake 3.17
# 3.8 for CUDA language
# 3.12 for add_compile_definitions
# 3.13 for modifying targets in subdirectories using add_subdirectory()
# 3.16 for OpenACC imported target
# 3.17 for FindCUDAToolkit
# 3.18 for BLAS imported target
cmake_minimum_required(VERSION 3.17)

# Note that the default settings for RelWithDebInfo will disable assertions
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE
    STRING "Choose the type of build (Debug Release RelWithDebInfo MinSizeRel)."
    FORCE
    )
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Hide symbols by default
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

set(PROJECT_NAME ExCALISparse)
set(LIB_NAME spmv)
set(PACKAGE_VERSION 1.0)
project(${PROJECT_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES CXX C)

# Set CMake behaviour
# Do not remove leading and trailing whitespace from linked libraries
cmake_policy(SET CMP0004 NEW)

# Set RPATH behaviour
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
# Embed absolute paths to external libraries that are not part of
# the project, (they are expected to be at the same location on all
# machines the project will be deployed to
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
include(GNUInstallDirs)
# Define RPATH for executables via a relative expression to enable a
# fully relocatable package
file(RELATIVE_PATH relDir1
 ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
 ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
file(RELATIVE_PATH relDir2
 ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/demos
 ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH $ORIGIN/${relDir1}:$ORIGIN/${relDir2})

option(ENABLE_OPENMP "Use OpenMP for shared-memory parallelization" OFF)
option(ENABLE_OPENMP_OFFLOAD "Use OpenMP for accelerator offloading" OFF)
option(ENABLE_OPENACC "Use OpenACC for accelerator offloading" OFF)
option(ENABLE_DPCPP "Use DPC++ for shared-memory parallelization" OFF)
option(ENABLE_HIPSYCL "Use hipSYCL for shared-memory parallelization" OFF)
option(ENABLE_CUDA "Use CUDA (cuBLAS, cuSPARSE) for GPU offloading" OFF)
option(ENABLE_MKL "Use Intel MKL to accelerate SpMV kernels whenever possible" OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(MPI REQUIRED COMPONENTS C CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  if(ENABLE_OPENMP)
    set(BLA_VENDOR Intel10_64lp)
  else()
    set(BLA_VENDOR Intel10_64lp_seq)
  endif()
endif()
find_package(BLAS REQUIRED)

if(ENABLE_OPENMP OR ENABLE_OPENMP_OFFLOAD)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    if("${OpenMP_CXX_LIBRARIES}" STREQUAL "")
      message(STATUS "No OpenMP library needs to be linked against")
    else()
      message(STATUS "Will link against OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")
    endif()
  else()
    message(WARNING "OpenMP was requested, but not found")
  endif()
endif()

if(ENABLE_OPENACC)
  find_package(OpenACC)
endif()

if(ENABLE_DPCPP)
  # Check compiler is set to dpcpp
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(FATAL_ERROR "Please set dpcpp as the C++ compiler")
  endif()
  find_package(MKL CONFIG REQUIRED)
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

if(ENABLE_HIPSYCL)
  find_package(hipSYCL CONFIG REQUIRED)
  find_package(CUDAToolkit)
  # We rely on the __HIPSYCL__ macro
  add_compile_options(-Wno-unknown-cuda-version -Wno-unknown-attributes)
endif()

if(ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  find_package(CUDAToolkit REQUIRED)
  # Support NVHPC toolkit
  # NVHPC id >= 3.20
  if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC" OR CMAKE_CXX_COMPILER_ID MATCHES "PGI")
    if(CMAKE_VERSION VERSION_LESS 3.22)
      # Find required CUDA math libraries explicitly
      find_library(CUBLAS_LIB NAMES cublas HINTS "${CUDAToolkit_ROOT}/../math_libs/lib64")
      if(CUBLAS_LIB STREQUAL CUBLAS_LIB-NOTFOUND)
	message(FATAL_ERROR "NVIDIA cublas library not found")
      endif()
      find_library(CUSPARSE_LIB cusparse PATHS "${CUDAToolkit_ROOT}/../math_libs/lib64")
      if(CUSPARSE_LIB STREQUAL CUSPARSE_LIB-NOTFOUND)
	message(FATAL_ERROR "NVIDIA cusparse library not found")
      endif()
    endif()
  endif()
endif()

# Generic compiler flags for development
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-forward-unknown-to-host-compiler>)
add_compile_options(-Wall -Wextra -pedantic -Wno-unused-parameter) # -Wno-unused-but-set-variable)

add_subdirectory(spmv)
include_directories(${LIB_NAME} ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${LIB_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
add_subdirectory(demos)
enable_testing()
add_subdirectory(tests)

# Require CMake 3.10
cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME spmv)
set(PACKAGE_VERSION 1.0)
project(${PROJECT_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES CXX)

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)

set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(MPI REQUIRED)

option(USE_OPENMP "Use OpenMP for shared-memory parallelization" OFF)
option(USE_DPCPP "Use DPC++ for shared-memory parallelization" OFF)
option(USE_MKL_SEQUENTIAL "Use sequential (non-threaded) Intel MKL to accelerate SpMV kernels whenever possible" OFF)
option(USE_MKL_PARALLEL "Use threaded Intel MKL to accelerate SpMV kernels whenever possible" OFF)

# Check that both OpenMP and DPC++ are not requested
if(USE_OPENMP AND USE_DPCPP)
  message(FATAL_ERROR "Both OpenMP and DPC++ were requested, which is not feasible.")
endif()

if(USE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    if("${OpenMP_CXX_LIBRARIES}" STREQUAL "")
      message(STATUS "No OpenMP library needs to be linked against")
    else()
      message(STATUS "Will link against OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")
    endif()
  else()
    message(WARNING "OpenMP was requested, but not found.")
  endif()
endif()

if(USE_DPCPP)
  if(DEFINED ENV{ONEAPI_ROOT})
    message(STATUS "Using DPC++ toolchain in ${ONEAPI_ROOT}")
  else()
    message(FATAL_ERROR "Please set ONEAPI_ROOT using the vendor-provided script to enable use of the DPC++ toolchain.")
  endif()
  # Force the C++ compiler to be dpcpp
  set(CMAKE_CXX_COMPILER dpcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SYCL")
  if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice")
  endif()
endif()

# Can use imported target BLAS::BLAS for version >= 3.18
if(USE_MKL_SEQUENTIAL)
  set(BLA_VENDOR Intel10_64lp_seq)
  find_package(BLAS)
  if(NOT BLAS_FOUND)
    message(WARNING "Intel MKL was requested, but not found.")
  endif()
endif()

# Can use imported target BLAS::BLAS for version >= 3.18
if(USE_MKL_PARALLEL)
  set(BLA_VENDOR Intel10_64lp)
  find_package(BLAS)
  if(NOT BLAS_FOUND)
    message(WARNING "Intel MKL was requested, but not found.")
  endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wnull-dereference")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_MAX_ALIGN_BYTES=32")

add_subdirectory(spmv)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(demos)
enable_testing()
add_subdirectory(tests)

# Require CMake 3.5
cmake_minimum_required(VERSION 3.5)

# Choose Release or Debug
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME spmv)
set(PACKAGE_VERSION 1.0)
project(${PROJECT_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES CXX)

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)

set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

find_package(MPI REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR} ${MPI_INCLUDE_PATH})

option(USE_OPENMP "Use OpenMP for shared-memory parallelization" OFF)
if(USE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        if("${OpenMP_CXX_LIBRARIES}" STREQUAL "")
            message(STATUS "No OpenMP library needs to be linked against")
        else()
            message(STATUS "Will link against OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")
        endif()
    else()
        message(WARNING "OpenMP was requested but support was not found")
    endif()
endif()

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)
if(BLAS_FOUND)
    include_directories("/usr/include/mkl")
endif()

# Not sure why need to set -O3 and -DNDEBUG as should be set by Release build type
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=all -g -O3 -DNDEBUG -DEIGEN_MAX_ALIGN_BYTES=32")

add_subdirectory(spmv)
include_directories (${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(demos)

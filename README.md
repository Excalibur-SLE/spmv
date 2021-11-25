# LIBSPMV

## Overview

LIBSPMV is a proof-of-concept distributed-memory Sparse Matrix-Vector Multiplication (SpMV) library. It currently implements pure-MPI and hybrid MPI+X where X is one of the following:
1. OpenMP for CPUs and GPUs
2. SYCL
3. CUDA
 The library also provides an implementation of the Conjugate Gradient (CG) method for solving large sparse linear systems.

## Getting Started

### Prerequisites
* CMake >= 3.17
* C++ compiler with C++17 support
* Eigen >= 3.3.9 (this dependency will be deprecated soon)
* MPI implementation that supports the MPI-3.0 standard and is CUDA-aware (in case USE_CUDA=on)

### Installation
It is recommended to build LIBSPMV in a separate directory form the source directory. The basic steps for building with CMake are:
1. Create a build directory, outside of the source directory.
2. In your build directory run `cmake <path-to-libspmv-src>` 
3. It is recommended to set options by calling `ccmake` in your build directory. Alternatively you can use the `-DVARIABLE=value` syntax in the previous step.
4. Run `make` to build.
5. Run `make test` to run sanity checks.
6. Run `make install` to install in directory defined through the `-DCMAKE_INSTALL_PREFIX=<path>` option.

#### Optional features
- To enable OpenMP on CPUs, use the `-DUSE_OPENMP=on` option.
- To enable OpenMP offloading on GPUs, use the `-DUSE_OPENMP_OFFLOAD=on` option.
- To enable SYCL, use the `-DUSE_DPCPP=on` option for Intel DPC++ or `-DUSE_HIPSYCL=on` option for HipSYCL. For hipSYCL, also set the HIPSYCL_TARGETS environment variable to select the target devices, e.g., 'omp;cuda:sm_xx'.
- To enable GPU offloading with CUDA, use the `-DUSE_CUDA=on` option with -DCMAKE_CUDA_ARCHITECTURES=<sm_xx> to select the target device's compute capability.
- To enable use of Intel MKL kernels where possible, use the `-DUSE_MKL_SEQUENTIAL=on` option for pure-MPI or the `-DUSE_MKL_PARALLEL=on` option for hybrid MPI+OpenMP.

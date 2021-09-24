# LIBSPMV

## Overview

LIBSPMV is a proof-of-concept distributed-memory Sparse Matrix-Vector Multiplication (SpMV) library. It currently supports pure-MPI, hybrid MPI+OpenMP and hybrid MPI+SYCL runs. Use of GPUs is only supported through SYCL. The library also includes an implementation of the Conjugate Gradient (CG) method.

## Getting Started

### Prerequisites
* CMake >= 3.10
* C++ compiler with C++17 support
* Eigen >= 3.3.9 (this dependency will be deprecated soon)
* MPI implementation that supports the MPI-3.0 standard

### Installation
It is recommended to build LIBSPMV in a separate directory form the source directory. The basic steps for building with CMake are:
1. Create a build directory, outside of the source directory.
2. In your build directory run `cmake <path-to-libspmv-src>` 
3. It is recommended to set options by calling `ccmake` in your build directory. Alternatively you can use the `-DVARIABLE=value` syntax in the previous step.
4. Run `make` to build.
5. Run `make check` to run sanity checks.
6. Run `make install` to install in directory defined through the `-DCMAKE_INSTALL_PREFIX=<path>` option.

#### Optional features
1. To enable OpenMP, use the `-DUSE_OPENMP=on` option.
2. To enable SYCL (currently only through Intel DPCPP), use the `-DUSE_DPCPP=on` option.
3. To enable use of Intel MKL kernels where possible, use the `-DUSE_MKL_SEQUENTIAL=on` option for pure-MPI or the `-DUSE_MKL_PARALLEL=on` option for hybrid MPI+OpenMP.

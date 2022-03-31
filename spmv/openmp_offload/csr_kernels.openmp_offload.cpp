// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "csr_kernels.h"
#include "omp_offload_executor.h"
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <unordered_set>

#ifdef _MKL
#include <mkl.h>
#endif // _MKL

namespace spmv
{

template <typename T>
void CSRSpMV<T>::init(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
                      int32_t* rowptr, int32_t* colind, T* values,
                      bool symmetric, const OmpOffloadExecutor& exec)
{
  _symmetric = symmetric;
}

template <typename T>
void CSRSpMV<T>::run(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
                     const int32_t* rowptr, const int32_t* colind,
                     const T* values, const T* diagonal, T alpha,
                     T* __restrict__ in, T beta, T* __restrict__ out,
                     const OmpOffloadExecutor& exec) const
{
  if (_symmetric && num_non_zeros > 0) {
    #pragma omp target teams distribute parallel for map(to                        \
                                                     : diagonal[:num_rows])    \
    map(tofrom                                                                 \
        : out[:num_rows])
    for (int32_t i = 0; i < num_rows; ++i) {
      out[i] = beta * out[i];
    }

    // For GCC compiler, teams+parallel map to warps/wavefronts and simd maps to
    // threads/work items CUDA kernel launched: dim={#teams,1,1},
    // blocks={#threads,warp_size,1}
    // For PGI compiler, teams+parallel map to thread blocks and simd is not
    // used Clang/LLVM does not implement simd
    #pragma omp target teams distribute parallel for			\
      map(to : rowptr[:num_rows + 1])					\
      map(to : colind[:num_non_zeros])					\
      map(to : values[:num_non_zeros])					\      
      map(to : diagonal[:num_rows])					\
      map(to : in[:num_cols])						\
      map(tofrom : out[:num_rows])
    for (int32_t i = 0; i < num_rows; ++i) {
      T sum = diagonal[i] * in[i];

      for (int32_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        int32_t col = colind[j];
        T val = values[j];
        sum += val * in[col];
        #pragma omp atomic
        out[col] += alpha * val * in[i];
      }

      #pragma omp atomic
      out[i] += alpha * sum;
    }
  } else if (_symmetric) {
    #pragma omp target teams distribute parallel for			\
      map(to : diagonal[:num_rows])					\
      map(to : in[:num_cols])						\
      map(tofrom : out[:num_rows])
    for (int32_t i = 0; i < num_rows; ++i) {
      out[i] = alpha * diagonal[i] * in[i] + beta * out[i];
    }
  } else {
    // For GCC compiler, teams+parallel map to warps/wavefronts and simd maps to
    // threads/work items CUDA kernel launched: dim={#teams,1,1},
    // blocks={#threads,warp_size,1}
    // For PGI compiler, teams+parallel map to thread blocks and simd is not
    // used Clang/LLVM does not implement simd
    #pragma omp target teams distribute parallel for			\
      map(to : rowptr[:num_rows + 1])					\
      map(to : colind[:num_non_zeros])					\
      map(to : values[:num_non_zeros])					\      
      map(to : in[:num_cols])						\
      map(tofrom : out[:num_rows])
    for (int32_t i = 0; i < num_rows; ++i) {
      T sum = 0.0;
      for (int32_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        sum += values[j] * in[colind[j]];
      }

      out[i] = alpha * sum + beta * out[i];
    }
  }
}

template <typename T>
void CSRSpMV<T>::finalize(const OmpOffloadExecutor& exec) const
{
}

} // namespace spmv

// Explicit instantiations
template class spmv::CSRSpMV<float>;
template class spmv::CSRSpMV<double>;

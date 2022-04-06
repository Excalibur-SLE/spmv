// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "csr_kernels.h"
#include "openacc_executor.h"

namespace spmv
{

template <typename T>
void CSRSpMV<T>::init(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
                      int32_t* rowptr, int32_t* colind, T* values,
                      bool symmetric, const OpenaccExecutor& exec)
{
  _symmetric = symmetric;
}

template <typename T>
void CSRSpMV<T>::run(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
                     const int32_t* rowptr, const int32_t* colind,
                     const T* values, const T* diagonal, T alpha,
                     T* __restrict__ in, T beta, T* __restrict__ out,
                     const OpenaccExecutor& exec) const
{
  if (_symmetric && num_non_zeros > 0) {
    #pragma acc kernels present(out)
    for (int32_t i = 0; i < num_rows; ++i) {
      out[i] = beta * out[i];
    }

    #pragma acc kernels				\
      present(rowptr)				\
      present(colind)				\
      present(values)				\
      present(diagonal)				\
      present(in)				\
      present(out)				
    for (int32_t i = 0; i < num_rows; ++i) {
      T sum = diagonal[i] * in[i];

      for (int32_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        int32_t col = colind[j];
        T val = values[j];
        sum += val * in[col];
        #pragma acc atomic
        out[col] += alpha * val * in[i];
      }

      #pragma acc atomic
      out[i] += alpha * sum;
    }
  } else if (_symmetric) {
    #pragma acc kernels present(in, out)
    for (int32_t i = 0; i < num_rows; ++i) {
      out[i] = alpha * diagonal[i] * in[i] + beta * out[i];
    }
  } else {
    #pragma acc kernels				\
      present(rowptr)				\
      present(colind)				\
      present(values)				\
      present(in)				\
      present(out)				
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
void CSRSpMV<T>::finalize(const OpenaccExecutor& exec) const
{
}

} // namespace spmv

// Explicit instantiations
template class spmv::CSRSpMV<float>;
template class spmv::CSRSpMV<double>;

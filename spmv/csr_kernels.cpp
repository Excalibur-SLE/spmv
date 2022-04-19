// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "csr_kernels.h"

namespace spmv
{

template <typename T>
void CSRSpMV<T>::init(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
                      int32_t* rowptr, int32_t* colind, T* values,
                      bool symmetric, const ReferenceExecutor& exec)
{
  _symmetric = symmetric;
}

template <typename T>
void CSRSpMV<T>::run(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
                     const int32_t* rowptr, const int32_t* colind,
                     const T* values, const T* diagonal, T alpha,
                     T* __restrict__ in, T beta, T* __restrict__ out,
                     const ReferenceExecutor& exec) const
{
  if (_symmetric) {
    for (int32_t i = 0; i < num_rows; ++i) {
      T sum = diagonal[i] * in[i];

      if (num_non_zeros > 0) {
        for (int32_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
          int col = colind[j];
          T val = values[j];
          sum += val * in[col];
          out[col] += alpha * val * in[i];
        }
      }

      out[i] = alpha * sum + beta * out[i];
    }
  } else {
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
void CSRSpMV<T>::finalize(const ReferenceExecutor& exec) const
{
}

} // namespace spmv

// Explicit instantiations
template class spmv::CSRSpMV<float>;
template class spmv::CSRSpMV<double>;

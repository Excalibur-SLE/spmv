// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "coo_kernels.h"

namespace spmv
{

template <typename T>
void COOSpMV<T>::init(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
                      int32_t* rowind, int32_t* colind, T* values,
                      const ReferenceExecutor& exec)
{
}

template <typename T>
void COOSpMV<T>::run(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
                     const int32_t* rowind, const int32_t* colind,
                     const T* values, T alpha, T* __restrict__ in, T beta,
                     T* __restrict__ out, const ReferenceExecutor& exec) const
{
  for (int32_t i = 0; i < num_non_zeros; ++i) {
    out[rowind[i]] += values[i] * in[colind[i]];
  }
}

template <typename T>
void COOSpMV<T>::finalize(const ReferenceExecutor& exec) const
{
}

} // namespace spmv

// Explicit instantiations
template class spmv::COOSpMV<float>;
template class spmv::COOSpMV<double>;

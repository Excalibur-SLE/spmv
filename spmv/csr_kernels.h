// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include <cstdlib>

namespace spmv
{

class ReferenceExecutor;
class OmpExecutor;
class OmpOffloadExecutor;
class SyclExecutor;
class CudaExecutor;

template <typename T>
class CSRSpMV
{
public:
  void init(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
            int32_t* rowptr, int32_t* colind, T* values, bool symmetric,
            const ReferenceExecutor& exec);
  void init(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
            int32_t* rowptr, int32_t* colind, T* values, bool symmetric,
            const OmpExecutor& exec);
  void init(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
            int32_t* rowptr, int32_t* colind, T* values, bool symmetric,
            const OmpOffloadExecutor& exec);
  void init(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
            int32_t* rowptr, int32_t* colind, T* values, bool symmetric,
            const SyclExecutor& exec);
  void init(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
            int32_t* rowptr, int32_t* colind, T* values, bool symmetric,
            const CudaExecutor& exec);
  void run(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
           const int32_t* rowptr, const int32_t* colind, const T* values,
           const T* diagonal, T alpha, T* __restrict__ in, T beta,
           T* __restrict__ out, const ReferenceExecutor& exec) const;
  void run(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
           const int32_t* rowptr, const int32_t* colind, const T* values,
           const T* diagonal, T alpha, T* __restrict__ in, T beta,
           T* __restrict__ out, const OmpExecutor& exec) const;
  void run(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
           const int32_t* rowptr, const int32_t* colind, const T* values,
           const T* diagonal, T alpha, T* __restrict__ in, T beta,
           T* __restrict__ out, const OmpOffloadExecutor& exec) const;
  void run(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
           const int32_t* rowptr, const int32_t* colind, const T* values,
           const T* diagonal, T alpha, T* __restrict__ in, T beta,
           T* __restrict__ out, const SyclExecutor& exec) const;
  void run(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
           const int32_t* rowptr, const int32_t* colind, const T* values,
           const T* diagonal, T alpha, T* __restrict__ in, T beta,
           T* __restrict__ out, const CudaExecutor& exec) const;
  void finalize(const ReferenceExecutor& exec) const;
  void finalize(const OmpExecutor& exec) const;
  void finalize(const OmpOffloadExecutor& exec) const;
  void finalize(const SyclExecutor& exec) const;
  void finalize(const CudaExecutor& exec) const;

private:
  bool _symmetric = false;
  void* _aux_data = nullptr;
};

} // namespace spmv

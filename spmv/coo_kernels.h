// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include "config.h"
#include "spmv_export.h"

#include <cstdlib>

namespace spmv
{

class ReferenceExecutor;
class CudaExecutor;

template <typename T>
class COOSpMV
{
public:
  void init(int32_t num_rows, int32_t num_cols, int64_t num_non_zeros,
            const int32_t* rowind, const int32_t* colind, const T* values,
            const ReferenceExecutor& exec);
  void run(int32_t num_rows, int32_t num_cols, int64_t num_non_zeros,
           const int32_t* rowind, const int32_t* colind, const T* values,
           T alpha, T* __restrict__ in, T beta, T* __restrict__ out,
           const ReferenceExecutor& exec) const;
  void finalize(const ReferenceExecutor& exec) const;

  void init(int32_t num_rows, int32_t num_cols, int64_t num_non_zeros,
            const int32_t* rowind, const int32_t* colind, const T* values,
            const CudaExecutor& exec);
  void run(int32_t num_rows, int32_t num_cols, int64_t num_non_zeros,
           const int32_t* rowind, const int32_t* colind, const T* values,
           T alpha, T* __restrict__ in, T beta, T* __restrict__ out,
           const CudaExecutor& exec) const;
  void finalize(const CudaExecutor& exec) const;

private:
  void* _aux_data = nullptr;
};

} // namespace spmv

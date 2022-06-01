// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include "config.h"
#include "spmv_export.h"

#include "coo_kernels.h"
#include "sub_matrix.h"

namespace spmv
{

// Forward declarations
class DeviceExecutor;

template <typename T>
class SPMV_EXPORT COOMatrix final : public SubMatrix<T>
{
public:
  COOMatrix(std::shared_ptr<DeviceExecutor> exec,
            const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat);
  COOMatrix(std::shared_ptr<DeviceExecutor> exec, int32_t num_rows,
            int32_t num_cols, int64_t num_non_zeros, const int32_t* rowptr,
            const int32_t* colind, const T* values);
  ~COOMatrix();

  size_t format_size() const override;

  void mult(T alpha, T* __restrict__ in, T beta,
            T* __restrict__ out) const override;

  // Extended API
  const int32_t* rowind() const { return _rowind; }
  const int32_t* colind() const { return _colind; }
  const T* values() const { return _values; }
  int32_t* rowind() { return _rowind; }
  int32_t* colind() { return _colind; }
  T* values() { return _values; }

private:
  int32_t* _rowind = nullptr;
  int32_t* _colind = nullptr;
  T* _values = nullptr;
  COOSpMV<T> _op;
};

/* // These should be hidden from spmv namespace */
/* void _rowptr2rowind(const ReferenceExecutor& exec, int32_t num_rows, */
/*                     int64_t num_non_zeros, const int32_t* rowptr, */
/*                     int32_t* rowind); */
/* #ifdef _CUDA */
/* void _rowptr2rowind(const CudaExecutor& exec, int32_t num_rows, */
/*                     int64_t num_non_zeros, const int32_t* rowptr, */
/*                     int32_t* rowind); */
/* #endif // _CUDA */

} // namespace spmv

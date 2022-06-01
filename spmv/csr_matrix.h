// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include "config.h"
#include "spmv_export.h"

#include "csr_kernels.h"
#include "sub_matrix.h"

namespace spmv
{

// Forward declarations
class DeviceExecutor;

template <typename T>
class SPMV_EXPORT CSRMatrix final : public SubMatrix<T>
{
public:
  CSRMatrix(std::shared_ptr<DeviceExecutor> exec,
            const Eigen::SparseMatrix<T, Eigen::RowMajor>* mat,
            const Eigen::Matrix<T, Eigen::Dynamic, 1>* diagonal = nullptr,
            bool symmetric = false);
  CSRMatrix(std::shared_ptr<DeviceExecutor> exec, int32_t num_rows,
            int32_t num_cols, int64_t num_non_zeros, const int32_t* rowptr,
            const int32_t* colind, const T* values, const T* diagonal = nullptr,
            bool symmetric = false);
  ~CSRMatrix();

  size_t format_size() const override;

  void mult(T alpha, T* __restrict__ in, T beta,
            T* __restrict__ out) const override;

  // Extended API
  const int32_t* rowptr() const { return _rowptr; }
  const int32_t* colind() const { return _colind; }
  const T* values() const { return _values; }
  int32_t* rowptr() { return _rowptr; }
  int32_t* colind() { return _colind; }
  T* values() { return _values; }

private:
  int32_t* _rowptr = nullptr;
  int32_t* _colind = nullptr;
  T* _values = nullptr;
  CSRSpMV<T> _op;
};

} // namespace spmv

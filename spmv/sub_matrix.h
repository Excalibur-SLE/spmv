// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include "config.h"
#include "spmv_export.h"

// Needed for hipSYCL
#ifdef __HIPSYCL__
#undef SYCL_DEVICE_ONLY
#endif
#include <Eigen/Sparse>
#include <memory>

namespace spmv
{

class DeviceExecutor;

// Abstract polymorphic base class for representing a sub-matrix
template <typename T>
class SubMatrix
{
public:
  SubMatrix() = default;
  SubMatrix(const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat,
            std::shared_ptr<DeviceExecutor> exec);
  SubMatrix(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
            const int32_t* rowptr, const int32_t* colind, const T* values,
            std::shared_ptr<DeviceExecutor> exec);
  virtual ~SubMatrix(){};

  /// Number of rows in the matrix
  int rows() const { return _num_rows; };

  /// Number of columns in the matrix
  int cols() const { return _num_cols; };

  /// Number of non-zeros in the matrix
  int non_zeros() const { return _num_non_zeros; };

  /// True if symmetry is used in matrix encoding
  bool symmetric() const { return _symmetric; }

  // Return diagonal of matrix
  const T* diagonal() const { return _diagonal; }
  T* diagonal() { return _diagonal; }

  /// The size of the matrix encoding in bytes
  virtual size_t format_size() const = 0;

  /// Multiplication with a dense vector
  virtual void mult(T alpha, T* __restrict__ in, T beta,
                    T* __restrict__ out) const = 0;

protected:
  std::shared_ptr<DeviceExecutor> _exec = nullptr;
  int _num_rows = 0;
  int _num_cols = 0;
  int _num_non_zeros = 0;
  bool _symmetric = false;
  T* _diagonal = nullptr;
};

} // namespace spmv

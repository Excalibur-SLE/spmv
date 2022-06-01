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

// Forward declarations
class DeviceExecutor;

/**
 * @brief Abstract polymorphic base class for representing a sparse sub-matrix
 */
template <typename T>
class SubMatrix
{
public:
  SubMatrix() = default;

  /**
   * Construct a sparse matrix from an Eigen matrix.
   * @param[in] exec Device executor
   * @param[in] mat Eigen matrix
   * @param[in] diagonal Matrix diagonal
   * @param[in] symmetric Flag to indicate whether to store half the matrix
   */
  SubMatrix(std::shared_ptr<DeviceExecutor> exec,
            const Eigen::SparseMatrix<T, Eigen::RowMajor>* mat,
            const Eigen::Matrix<T, Eigen::Dynamic, 1>* diagonal = nullptr,
            bool symmetric = false);

  /**
   * Construct a sparse matrix from a CSR matrix.
   * @param[in] exec Device executor
   * @param[in] num_rows Number of rows in the matrix
   * @param[in] num_cols Number of columns in the matrix
   * @param[in] num_non_zeros Number of non-zeros in the matrix
   * @param[in] rowptr Array of indices to the start of each row in the
   * colind/values arrays
   * @param[in] colind Array containing column indices for each non-zero element
   * in the matrix
   * @param[in] values Array containing values of each non-zero element in the
   * matrix
   * @param[in] diagonal Array of main diagonal values
   * @param[in] symmetric Flag to indicate whether to store half the matrix
   */
  SubMatrix(std::shared_ptr<DeviceExecutor> exec, int32_t num_rows,
            int32_t num_cols, int64_t num_non_zeros, const int32_t* rowptr,
            const int32_t* colind, const T* values, const T* diagonal = nullptr,
            bool symmetric = false);

  /**
   * Destructor
   */
  virtual ~SubMatrix(){};

  /**
   * Returns the number of rows in the matrix.
   * @return The number of rows in the matrix
   */
  int32_t rows() const { return _num_rows; };

  /**
   * Returns the number of columns in the matrix.
   * @return The number of columns in the matrix
   */
  int32_t cols() const { return _num_cols; };

  /**
   * Returns the number of non-zeros in the matrix.
   * @return The number of non-zeros in the matrix
   */
  int64_t non_zeros() const { return _num_non_zeros; };

  /**
   * Indicates whether symmetry is used in matrix encoding.
   * @return True if half the matrix is stored
   */
  bool symmetric() const { return _symmetric; }

  /**
   * Returns diagonal of matrix.
   * @return The number of non-zeros in the matrix
   */
  const T* diagonal() const { return _diagonal; }
  T* diagonal() { return _diagonal; }

  /**
   * Returns the size of the matrix encoding in bytes
   * @return The size of the matrix encoding in bytes
   */
  virtual size_t format_size() const = 0;

  /**
   * Multiplies the matrix with a dense vector, y = alpha*A*x + beta*y
   * @param[in] alpha Scalar parameter
   * @param[in] in Input vector array
   * @param[in] beta Scalar parameter
   * @param[in,out] out Output vector array (should be initialized)
   */
  virtual void mult(T alpha, T* __restrict__ in, T beta,
                    T* __restrict__ out) const = 0;

protected:
  std::shared_ptr<DeviceExecutor> _exec = nullptr;
  int32_t _num_rows = 0;
  int32_t _num_cols = 0;
  int64_t _num_non_zeros = 0;
  bool _symmetric = false;
  T* _diagonal = nullptr;
};

} // namespace spmv

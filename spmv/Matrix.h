// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include "config.h"
#include "spmv_export.h"

// Needed for interoperability of hipSYCL with Eigen
#ifdef __HIPSYCL__
#undef SYCL_DEVICE_ONLY
#endif // _HIPSYCL
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <mpi.h>
#include <vector>

#include "mpi_utils.h"

using namespace std;

namespace spmv
{

// Forward declarations
template <typename T>
class SubMatrix;
class DeviceExecutor;
class L2GMap;

template <typename T>
class SPMV_EXPORT Matrix
{
  /// Matrix with row and column maps.
public:
  /// This constructor just copies in the data. To build a Matrix from more
  /// general data, use `Matrix::create_matrix` instead.
  Matrix(const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat,
         shared_ptr<spmv::L2GMap> col_map, shared_ptr<spmv::L2GMap> row_map,
         std::shared_ptr<DeviceExecutor> exec);

  /// This constructor just copies in the data from the "local" and "remote"
  /// sub-blocks of a matrix. To build a Matrix from more
  /// general data, use `Matrix::create_matrix` instead.
  Matrix(const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat_local,
         const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat_remote,
         shared_ptr<spmv::L2GMap> col_map, shared_ptr<spmv::L2GMap> row_map,
         std::shared_ptr<DeviceExecutor> exec);

  /// This constructor just copies in the data from the "local" and "remote"
  /// sub-blocks of a symmetric matrix. To build a Matrix from more
  /// general data, use `Matrix::create_matrix` instead.
  Matrix(const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat_local,
         const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat_remote,
         const Eigen::Matrix<T, Eigen::Dynamic, 1>& mat_diagonal,
         shared_ptr<spmv::L2GMap> col_map, shared_ptr<spmv::L2GMap> row_map,
         int nnz_full, std::shared_ptr<DeviceExecutor> exec);

  /// Destructor
  ~Matrix();

  /// Number of rows in the matrix
  int rows() const;
  /// Number of columns in the matrix
  int cols() const;
  /// Number of non-zeros in the matrix
  int non_zeros() const;
  /// True if symmetry is used in matrix encoding
  bool symmetric() const { return _symmetric; }
  /// The size of the matrix encoding in bytes
  size_t format_size() const;

  /// MatVec operator
  /// Interface using Eigen vectors
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  mult(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const;
  // Interface using pointers
  void mult(T* __restrict__ x, T* __restrict__ y) const;

  /// MatVec operator for A^T x
  /// Normal interface using Eigen vectors
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  transpmult(const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;

  /// Row mapping (local-to-global). Usually, there will not be ghost rows.
  shared_ptr<L2GMap> row_map() const { return _row_map; }

  /// Column mapping (local-to-global)
  shared_ptr<const L2GMap> col_map() const { return _col_map; }

  /// Create an `spmv::Matrix` from a CSR matrix and row and column
  /// mappings, such that the resulting matrix has no row ghosts, but only
  /// column ghosts. This is achieved by sending ghost rows to their owners,
  /// where they are summed into existing rows. The column ghost mapping will
  /// also change in this process.
  static Matrix<T>* create_matrix(
      MPI_Comm comm, std::shared_ptr<DeviceExecutor> exec,
      const Eigen::SparseMatrix<T, Eigen::RowMajor> mat, int64_t nrows_local,
      int64_t ncols_local, vector<int64_t> row_ghosts,
      vector<int64_t> col_ghosts, bool symmetric = false,
      CommunicationModel cm = CommunicationModel::collective_blocking);

  /// Create an `spmv::Matrix` from a CSR matrix and row and column
  /// mappings, such that the resulting matrix has no row ghosts, but only
  /// column ghosts. This is achieved by sending ghost rows to their owners,
  /// where they are summed into existing rows. The column ghost mapping will
  /// also change in this process.
  static Matrix<T>* create_matrix(
      MPI_Comm comm, std::shared_ptr<DeviceExecutor> exec,
      const int32_t* rowptr, const int32_t* colind, const T* values,
      int64_t nrows_local, int64_t ncols_local, vector<int64_t> row_ghosts,
      vector<int64_t> col_ghosts, bool symmetric = false,
      CommunicationModel cm = CommunicationModel::collective_blocking);

private:
  std::shared_ptr<DeviceExecutor> _exec = nullptr;
  // Storage for matrix
  std::unique_ptr<SubMatrix<T>> _mat_local = nullptr;
  std::unique_ptr<SubMatrix<T>> _mat_remote = nullptr;
  // Column and Row maps: usually _row_map will not have ghosts
  shared_ptr<spmv::L2GMap> _col_map = nullptr;
  shared_ptr<spmv::L2GMap> _row_map = nullptr;
  // Auxiliary data
  int _nnz = 0;
  bool _symmetric = false;

  /// Vanilla SpMV kernel
  void spmv(T* x, T* y) const;
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  spmv(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const;
  /// SpMV kernel with comm/comp overlap
  void spmv_overlap(T* x, T* y) const;
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  spmv_overlap(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const;
  /// Symmetric SpMV kernel
  void spmv_sym(T* x, T* y) const;
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  spmv_sym(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const;
  /// Symmetric SpMV kernel with comm/comp overlap
  void spmv_sym_overlap(T* x, T* y) const;
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  spmv_sym_overlap(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const;
};
} // namespace spmv

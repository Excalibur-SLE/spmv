// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <vector>

#ifdef EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif

#include <mpi.h>

#pragma once

/// Simple Distributed Sparse Linear Algebra Library
namespace spmv
{

class L2GMap;

template <typename T>
class Matrix
{
  /// Matrix with row and column maps.
public:
  /// This constructor just shares the data. To build a Matrix from more
  /// general data, use `Matrix::create_matrix` instead.
  Matrix(std::shared_ptr<Eigen::SparseMatrix<T, Eigen::RowMajor>> A,
         std::shared_ptr<spmv::L2GMap> col_map,
         std::shared_ptr<spmv::L2GMap> row_map);

  /// Destructor (destroys MKL structs, if using MKL)
  ~Matrix();

  /// MatVec operator for A x
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  operator*(const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;

  /// MatVec operator for A^T x
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  transpmult(const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;

  /// Row mapping (local-to-global). Usually, there will not be ghost rows.
  std::shared_ptr<const L2GMap> row_map() const { return _row_map; }

  /// Column mapping (local-to-global)
  std::shared_ptr<const L2GMap> col_map() const { return _col_map; }

  /// Access the underlying local sparse matrix
  Eigen::SparseMatrix<T, Eigen::RowMajor>& mat() { return *_matA; }
  const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat() const { return *_matA; }

  /// Create an `spmv::Matrix` from an Eigen::SparseMatrix and row and column
  /// mappings, such that the resulting matrix has no row ghosts, but only
  /// column ghosts. This is achieved by sending ghost rows to their owners,
  /// where they are summed into existing rows. The column ghost mapping will
  /// also change in this process.
  static Matrix<T>
  create_matrix(MPI_Comm comm,
                const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat,
                std::int64_t nrows_local, std::int64_t ncols_local,
                std::vector<std::int64_t> row_ghosts,
                std::vector<std::int64_t> col_ghosts);

private:
// MKL pointers to Eigen data
#ifdef EIGEN_USE_MKL_ALL
  sparse_matrix_t A_mkl;
  struct matrix_descr mat_desc;
  void mkl_init();
#endif

  // Storage for Matrix
  std::shared_ptr<Eigen::SparseMatrix<T, Eigen::RowMajor>> _matA;

  // Column and Row maps: usually _row_map will not have ghosts.
  std::shared_ptr<spmv::L2GMap> _col_map;
  std::shared_ptr<spmv::L2GMap> _row_map;
};
} // namespace spmv

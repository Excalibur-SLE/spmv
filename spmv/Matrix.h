// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
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
  /// This constructor just copies in the data. To build a Matrix from more
  /// general data, use `Matrix::create_matrix` instead.
  Matrix(std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat,
         std::shared_ptr<spmv::L2GMap> col_map,
         std::shared_ptr<spmv::L2GMap> row_map);

  /// This constructor just copies in the data from the "local" and "remote"
  /// sub-blocks of a matrix. To build a Matrix from more
  /// general data, use `Matrix::create_matrix` instead.
  Matrix(
      std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat_local,
      std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat_remote,
      std::shared_ptr<spmv::L2GMap> col_map,
      std::shared_ptr<spmv::L2GMap> row_map);

  /// This constructor just copies in the data from the "local" and "remote"
  /// sub-blocks of a symmetric matrix. To build a Matrix from more
  /// general data, use `Matrix::create_matrix` instead.
  Matrix(
      std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat_local,
      std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat_remote,
      std::shared_ptr<const Eigen::Matrix<T, Eigen::Dynamic, 1>> mat_diagonal,
      std::shared_ptr<spmv::L2GMap> col_map,
      std::shared_ptr<spmv::L2GMap> row_map, int nnz_full, bool overlap);

  /// Destructor
  ~Matrix();

  /// Number of rows in the matrix
  int rows() const { return _mat_local->rows(); }
  /// Number of columns in the matrix
  int cols() const { return _mat_local->cols(); }
  /// Number of non-zeros in the matrix
  int non_zeros() const
  {
    if (_symmetric)
      return _nnz;
    else if (_overlap)
      return _mat_local->nonZeros() + _mat_remote->nonZeros();
    else
      return _mat_local->nonZeros();
  }

  /// The size of the matrix encoding in bytes
  size_t format_size() const;

  /// MatVec operator for A x
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  operator*(Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;

  /// MatVec operator for A^T x
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  transpmult(const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;

  /// Row mapping (local-to-global). Usually, there will not be ghost rows.
  std::shared_ptr<L2GMap> row_map() const { return _row_map; }

  /// Column mapping (local-to-global)
  std::shared_ptr<const L2GMap> col_map() const { return _col_map; }

  /// Create an `spmv::Matrix` from an Eigen::SparseMatrix and row and column
  /// mappings, such that the resulting matrix has no row ghosts, but only
  /// column ghosts. This is achieved by sending ghost rows to their owners,
  /// where they are summed into existing rows. The column ghost mapping will
  /// also change in this process.
  static Matrix<T>
  create_matrix(MPI_Comm comm,
                const Eigen::SparseMatrix<T, Eigen::RowMajor> mat,
                std::int64_t nrows_local, std::int64_t ncols_local,
                std::vector<std::int64_t> row_ghosts,
                std::vector<std::int64_t> col_ghosts, bool symmetric = false,
                bool p2p = false, bool overlap = false);

  /// Create an `spmv::Matrix` from a CSR matrix and row and column
  /// mappings, such that the resulting matrix has no row ghosts, but only
  /// column ghosts. This is achieved by sending ghost rows to their owners,
  /// where they are summed into existing rows. The column ghost mapping will
  /// also change in this process.
  static Matrix<T> create_matrix(MPI_Comm comm, const std::int32_t* rowptr,
                                 const std::int32_t* colind, const T* values,
                                 std::int64_t nrows_local,
                                 std::int64_t ncols_local,
                                 std::vector<std::int64_t> row_ghosts,
                                 std::vector<std::int64_t> col_ghosts,
                                 bool symmetric = false, bool p2p = false,
                                 bool overlap = false);

private:
  // Storage for matrix
  std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> _mat_local;
  std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> _mat_remote;
  std::shared_ptr<const Eigen::Matrix<T, Eigen::Dynamic, 1>> _mat_diagonal;
#ifdef EIGEN_USE_MKL_ALL
  sparse_matrix_t _mat_local_mkl;
  sparse_matrix_t _mat_remote_mkl;
  struct matrix_descr _mat_local_desc;
  struct matrix_descr _mat_remote_desc;
#endif

  // Column and Row maps: usually _row_map will not have ghosts.
  std::shared_ptr<spmv::L2GMap> _col_map;
  std::shared_ptr<spmv::L2GMap> _row_map;

  // Auxiliary data
  int _nnz;
  bool _symmetric;
  // Flag to indicate whether to overlap ghosts update phase with computation
  bool _overlap;

#ifdef _OPENMP
  struct ConflictMap
  {
    int length;
    int* pos;
    short* tid;

    ConflictMap(const int ncnfls) : length(ncnfls)
    {
      pos = new int[ncnfls];
      tid = new short[ncnfls];
    }

    ~ConflictMap()
    {
      delete[] pos;
      delete[] tid;
    }
  };

  int _nthreads;
  ConflictMap* _cnfl_map;
  int* _row_split;
  int* _map_start;
  int* _map_end;
  T** _y_local;
#endif

  // Private helper functions
#ifdef _OPENMP
  /// Partition the matrix to threads so that every thread has approximately the
  /// same number of rows
  void partition_by_nrows(const int nthreads);
  /// Partition the matrix to threads so that every thread has approximately the
  /// same number of non-zeros
  void partition_by_nnz(const int nthreads);
  /// Tune the matrix for a number of threads. Can be called multiple times.
  void tune(const int nthreads);
#endif

  /// SpMV kernel with comm/comp overlap
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  spmv_overlap(Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;
  /// Symmetric SpMV kernel
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  spmv_sym(const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;
  /// Symmetric SpMV kernel with comm/comp overlap
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  spmv_sym_overlap(Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;

#ifdef EIGEN_USE_MKL_ALL
  /// Setup the Intel MKL library
  void mkl_init();
#endif
};
} // namespace spmv

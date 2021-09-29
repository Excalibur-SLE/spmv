// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "mpi_types.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <mpi.h>
#include <vector>

#ifdef USE_MKL
#include <mkl.h>
#endif // USE_MKL

#ifdef _SYCL
#include <CL/sycl.hpp>
using namespace sycl;
#endif // _SYCL

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
      std::shared_ptr<spmv::L2GMap> row_map, int nnz_full);

  /// Destructor
  ~Matrix();

  /// Number of rows in the matrix
  int rows() const { return _mat_local->rows(); }
  /// Number of columns in the matrix
  int cols() const { return _mat_local->cols(); }
  /// Number of non-zeros in the matrix
  int non_zeros() const;
  /// True if symmetry is used in matrix encoding
  bool symmetric() const { return _symmetric; }
  /// The size of the matrix encoding in bytes
  size_t format_size() const;

  /// MatVec operator
#ifdef _SYCL
  /// SYCL interface Using USM pointers
  void mult(queue& q, T* __restrict__ b, T* __restrict__ y) const;
#else
  /// Normal interface using Eigen vectors
  Eigen::Matrix<T, Eigen::Dynamic, 1>
      mult(Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;
#endif // _SYCL

  /// MatVec operator for A^T x
#ifdef _SYCL
  // FIXME
#else
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  transpmult(const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;
#endif // _SYCL

  /// Row mapping (local-to-global). Usually, there will not be ghost rows.
  std::shared_ptr<L2GMap> row_map() const { return _row_map; }

  /// Column mapping (local-to-global)
  std::shared_ptr<const L2GMap> col_map() const { return _col_map; }

  static Matrix<T>* create_matrix(
      MPI_Comm comm, const Eigen::SparseMatrix<T, Eigen::RowMajor> mat,
      std::int64_t nrows_local, std::int64_t ncols_local,
      std::vector<std::int64_t> row_ghosts,
      std::vector<std::int64_t> col_ghosts, bool symmetric = false,
      CommunicationModel cm = CommunicationModel::collective_blocking);

  /// Create an `spmv::Matrix` from a CSR matrix and row and column
  /// mappings, such that the resulting matrix has no row ghosts, but only
  /// column ghosts. This is achieved by sending ghost rows to their owners,
  /// where they are summed into existing rows. The column ghost mapping will
  /// also change in this process.
  static Matrix<T>* create_matrix(
      MPI_Comm comm, const std::int32_t* rowptr, const std::int32_t* colind,
      const T* values, std::int64_t nrows_local, std::int64_t ncols_local,
      std::vector<std::int64_t> row_ghosts,
      std::vector<std::int64_t> col_ghosts, bool symmetric = false,
      CommunicationModel cm = CommunicationModel::collective_blocking);

private:
  // Storage for matrix
  std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> _mat_local
      = nullptr;
  std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> _mat_remote
      = nullptr;
  std::shared_ptr<const Eigen::Matrix<T, Eigen::Dynamic, 1>> _mat_diagonal
      = nullptr;
#ifdef USE_MKL
  sparse_matrix_t _mat_local_mkl;
  sparse_matrix_t _mat_remote_mkl;
  struct matrix_descr _mat_local_desc;
  struct matrix_descr _mat_remote_desc;
#endif // USE_MKL

  // Column and Row maps: usually _row_map will not have ghosts.
  std::shared_ptr<spmv::L2GMap> _col_map = nullptr;
  std::shared_ptr<spmv::L2GMap> _row_map = nullptr;

  // Auxiliary data
  int _nnz = 0;
  bool _symmetric = false;

#if defined(_OPENMP) || defined(_SYCL)
  struct ConflictMap
  {
    int length = 0;
    int* pos = nullptr;
    short* vid = nullptr;

    ConflictMap(const int ncnfls) : length(ncnfls)
    {
      pos = new int[ncnfls];
      vid = new short[ncnfls];
    }

    ~ConflictMap()
    {
      delete[] pos;
      delete[] vid;
    }
  };

  int _nthreads = 1;
  int _ncnfls = 0;
  ConflictMap* _cnfl_map = nullptr;
  int* _row_split = nullptr;
  int* _map_start = nullptr;
  int* _map_end = nullptr;
  T** _y_local = nullptr;
#endif

#ifdef _SYCL
  // SYCL-specific auxiliary data
  buffer<int>* _d_rowptr_local = nullptr;
  buffer<int>* _d_colind_local = nullptr;
  buffer<T>* _d_values_local = nullptr;
  buffer<int>* _d_rowptr_remote = nullptr;
  buffer<int>* _d_colind_remote = nullptr;
  buffer<T>* _d_values_remote = nullptr;
  buffer<T>* _d_diagonal = nullptr;
  buffer<int>* _d_row_split = nullptr;
  buffer<int>* _d_map_start = nullptr;
  buffer<int>* _d_map_end = nullptr;
  buffer<short>* _d_cnfl_vid = nullptr;
  buffer<int>* _d_cnfl_pos = nullptr;
  buffer<T, 2>* _d_y_local = nullptr;
#endif // _SYCL

  // Private helper functions
#if defined(_OPENMP) || defined(_SYCL)
  /// Partition the matrix to threads so that every thread has approximately the
  /// same number of rows
  void partition_by_nrows(const int nthreads);
  /// Partition the matrix to threads so that every thread has approximately the
  /// same number of non-zeros
  void partition_by_nnz(const int nthreads);
  /// Tune the matrix for a number of threads. Can be called multiple times.
  void tune(const int nthreads);
#endif // _OPENMP || _SYCL

  /// SpMV kernel with comm/comp overlap
  Eigen::Matrix<T, Eigen::Dynamic, 1>
      spmv_overlap(Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;
  /// Symmetric SpMV kernel
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  spmv_sym(const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;
  /// Symmetric SpMV kernel with comm/comp overlap
  Eigen::Matrix<T, Eigen::Dynamic, 1>
      spmv_sym_overlap(Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;
#ifdef _SYCL
  /// SpMV kernel with USM pointers
  void spmv_sycl(queue& q, T* __restrict__ b, T* __restrict__ y) const;
  /// SpMV symmetric kernel with USM pointers
  void spmv_sym_sycl(queue& q, T* __restrict__ b, T* __restrict__ y) const;
#endif // _SYCL

#ifdef USE_MKL
  /// Setup the Intel MKL library
  void mkl_init();
#endif // USE_MKL
};
} // namespace spmv

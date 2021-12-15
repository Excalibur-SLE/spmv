// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once
// Needed for hipSYCL
#ifdef __HIPSYCL__
#undef SYCL_DEVICE_ONLY
#endif // _HIPSYCL
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <mpi.h>
#include <vector>

#ifdef _SYCL
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;
#endif // _SYCL

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif // USE_MKL

#include "mpi_types.h"

using namespace std;

/// Simple Distributed Sparse Linear Algebra Library
namespace spmv
{

// Forward declarations
class L2GMap;

// FIXME
struct MergeCoordinate {
  int row_idx;
  int val_idx;
};

#if defined(_OPENMP_OFFLOAD) || defined(_SYCL)
constexpr int TEAM_SIZE = 256;
constexpr int BLOCK_SIZE = 96;
constexpr int FACTOR = 1;
#endif

template <typename T>
class Matrix
{
  /// Matrix with row and column maps.
public:
  /// This constructor just copies in the data. To build a Matrix from more
  /// general data, use `Matrix::create_matrix` instead.
  Matrix(shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat,
         shared_ptr<spmv::L2GMap> col_map, shared_ptr<spmv::L2GMap> row_map);

  /// This constructor just copies in the data from the "local" and "remote"
  /// sub-blocks of a matrix. To build a Matrix from more
  /// general data, use `Matrix::create_matrix` instead.
  Matrix(shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat_local,
         shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat_remote,
         shared_ptr<spmv::L2GMap> col_map, shared_ptr<spmv::L2GMap> row_map);

  /// This constructor just copies in the data from the "local" and "remote"
  /// sub-blocks of a symmetric matrix. To build a Matrix from more
  /// general data, use `Matrix::create_matrix` instead.
  Matrix(shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat_local,
         shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat_remote,
         shared_ptr<const Eigen::Matrix<T, Eigen::Dynamic, 1>> mat_diagonal,
         shared_ptr<spmv::L2GMap> col_map, shared_ptr<spmv::L2GMap> row_map,
         int nnz_full);

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

#ifdef _SYCL
  /// Tune SpMV for a particular device
  void tune(sycl::queue& q);
#endif // _SYCL

  /// MatVec operator
  /// Interface using Eigen vectors
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  mult(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const;

#ifdef _OPENMP_OFFLOAD
  // Interface using normal pointers
  void mult(T* x, T* y) const;
#endif // _OPENMP_OFFLOAD

#ifdef _SYCL
  /// SYCL interface using buffers
  void mult(sycl::queue& q, sycl::buffer<T>& x_buf,
            sycl::buffer<T>& y_buf) const;
#endif // _SYCL

#ifdef USE_CUDA
  // Interface using CUDA device pointers
  void mult(T* __restrict__ x, T* __restrict__ y, cudaStream_t& stream) const;
#endif

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
      MPI_Comm comm, const Eigen::SparseMatrix<T, Eigen::RowMajor> mat,
      int64_t nrows_local, int64_t ncols_local, vector<int64_t> row_ghosts,
      vector<int64_t> col_ghosts, bool symmetric = false,
#ifdef USE_CUDA
      CommunicationModel cm = CommunicationModel::p2p_blocking);
#else
      CommunicationModel cm = CommunicationModel::collective_blocking);
#endif

  /// Create an `spmv::Matrix` from a CSR matrix and row and column
  /// mappings, such that the resulting matrix has no row ghosts, but only
  /// column ghosts. This is achieved by sending ghost rows to their owners,
  /// where they are summed into existing rows. The column ghost mapping will
  /// also change in this process.
  static Matrix<T>* create_matrix(MPI_Comm comm, const int32_t* rowptr,
                                  const int32_t* colind, const T* values,
                                  int64_t nrows_local, int64_t ncols_local,
                                  vector<int64_t> row_ghosts,
                                  vector<int64_t> col_ghosts,
                                  bool symmetric = false,
                                  CommunicationModel cm
#ifdef USE_CUDA
                                  = CommunicationModel::p2p_blocking);
#else
                                  = CommunicationModel::collective_blocking);
#endif

private:
  // Storage for matrix
  shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> _mat_local
      = nullptr;
  shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> _mat_remote
      = nullptr;
  shared_ptr<const Eigen::Matrix<T, Eigen::Dynamic, 1>> _mat_diagonal = nullptr;

  // Column and Row maps: usually _row_map will not have ghosts
  shared_ptr<spmv::L2GMap> _col_map = nullptr;
  shared_ptr<spmv::L2GMap> _row_map = nullptr;

  // Auxiliary data
  int _nnz = 0;
  bool _symmetric = false;

#if defined(_OPENMP) || defined(_SYCL)
  struct ConflictMap {
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
  int _nblocks = 1;
  int _ncnfls = 0;
  ConflictMap* _cnfl_map = nullptr;
  int* _row_split = nullptr;
  int* _map_start = nullptr;
  int* _map_end = nullptr;
#ifdef _SYCL
  T** _y_local = nullptr;
#else
  T* _y_local = nullptr;
#endif
#endif // _OPENMP || _SYCL

#ifdef _SYCL
  sycl::buffer<int>* _d_rowptr_local = nullptr;
  sycl::buffer<int>* _d_colind_local = nullptr;
  sycl::buffer<T>* _d_values_local = nullptr;
  sycl::buffer<int>* _d_rowptr_remote = nullptr;
  sycl::buffer<int>* _d_colind_remote = nullptr;
  sycl::buffer<T>* _d_values_remote = nullptr;
  sycl::buffer<T>* _d_diagonal = nullptr;
  sycl::buffer<int>* _d_row_split = nullptr;
  sycl::buffer<int>* _d_map_start = nullptr;
  sycl::buffer<int>* _d_map_end = nullptr;
  sycl::buffer<short>* _d_cnfl_vid = nullptr;
  sycl::buffer<int>* _d_cnfl_pos = nullptr;
  sycl::buffer<T, 2>* _d_y_local = nullptr;
  MergeCoordinate* _merge_path = nullptr;
  sycl::buffer<MergeCoordinate>* _d_merge_path = nullptr;
  sycl::buffer<int>* _d_carry_row = nullptr;
  sycl::buffer<T>* _d_carry_val = nullptr;
  int* _carry_row = nullptr;
  T* _carry_val = nullptr;
#endif // _SYCL

#ifdef USE_CUDA
  cusparseHandle_t _cusparse_handle = 0;
  cusparseSpMatDescr_t _mat_local_cusparse = nullptr;
  cusparseSpMatDescr_t _mat_remote_cusparse = nullptr;
  cusparseMatDescr_t _mat_local_desc = 0;
  cusparseMatDescr_t _mat_remote_desc = 0;
  int* _d_rowptr_local = nullptr;
  int* _d_colind_local = nullptr;
  T* _d_values_local = nullptr;
  T* _d_diagonal = nullptr;
  int* _d_rowptr_remote = nullptr;
  int* _d_rowind_remote = nullptr;
  int* _d_colind_remote = nullptr;
  T* _d_values_remote = nullptr;
  mutable void* _buffer = nullptr;
  mutable void* _buffer_rmt = nullptr;
#endif // USE_CUDA

#ifdef USE_MKL
  sparse_matrix_t _mat_local_mkl;
  sparse_matrix_t _mat_remote_mkl;
  struct matrix_descr _mat_local_desc;
  struct matrix_descr _mat_remote_desc;
#endif // USE_MKL

  // Private helper functions
#if defined(_OPENMP) || defined(_SYCL)
  /// Partition the matrix so that every compute unit has approximately the same
  /// number of rows
  void partition_by_nrows(const int ncus);
  /// Partition the matrix so that every compute unit has approximately the same
  /// number of non-zeros
  void partition_by_nnz(const int ncus);
  /// Tune the matrix for a number of compute units. Can be called multiple
  /// times.
  void tune_internal(const int ncus);
#endif // _OPENMP || _SYCL

  /// SpMV kernel with comm/comp overlap
  void spmv_overlap(T* x, T* y) const;
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  spmv_overlap(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const;
  /// Symmetric SpMV kernel
  void spmv_sym(const T* x, T* y) const;
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  spmv_sym(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const;
  /// Symmetric SpMV kernel with comm/comp overlap
  void spmv_sym_overlap(T* x, T* y) const;
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  spmv_sym_overlap(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const;

#ifdef _OPENMP_OFFLOAD
  /// Vanilla SpMV kernel for GPU accelerators
  void spmv_openmp_offload(const T* x, T* y) const;
  /// Symmetric SpMV kernel for GPU accelerators
  void spmv_sym_openmp_offload(const T* x, T* y) const;
#endif // _OPENMP_OFFLOAD

#ifdef _SYCL
  /// SpMV kernel with buffers
  void spmv_sycl(sycl::queue& q, sycl::buffer<T>& x_buf,
                 sycl::buffer<T>& y_buf) const;
  /// SpMV symmetric kernel with buffers
  void spmv_sym_sycl(sycl::queue& q, sycl::buffer<T>& x_buf,
                     sycl::buffer<T>& y_buf) const;
#endif // _SYCL

#ifdef USE_CUDA
  /// Setup the NVIDIA cuSPARSE library
  void cuda_init();
  void cuda_destroy();
  /// Symmetric SpMV kernel
  void spmv_sym_cuda(const T* __restrict__ x, T* __restrict__ y,
                     cudaStream_t& stream) const;
  void spmv_cuda(const T* __restrict__ x, T* __restrict__ y,
                 cudaStream_t& stream) const;
#endif // USE_CUDA

#ifdef USE_MKL
  /// Setup the Intel MKL library
  void mkl_init();
#endif // USE_MKL
};
} // namespace spmv

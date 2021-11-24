// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "L2GMap.h"
#include "Matrix.h"
#include "helper_cuda.h"

using namespace spmv;

//-----------------------------------------------------------------------------
template <>
void Matrix<float>::cusparse_init()
{
  // Get handle to the CUSPARSE context
  CHECK_CUSPARSE(cusparseCreate(&_cusparse_handle));

  // Create cuSPARSE CSR wrapper for local block
  CHECK_CUSPARSE(cusparseCreateCsr(
      &_mat_local_cusparse, _mat_local->rows(), _mat_local->cols(),
      _mat_local->nonZeros(), _d_rowptr_local, _d_colind_local, _d_values_local,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&_mat_local_desc));
  CHECK_CUSPARSE(
      cusparseSetMatType(_mat_local_desc, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE(
      cusparseSetMatIndexBase(_mat_local_desc, CUSPARSE_INDEX_BASE_ZERO));

  // Create cuSPARSE CSR wrapper for remote block if applicable
  if (_mat_remote != nullptr) {
    CHECK_CUSPARSE(cusparseCreateCsr(
        &_mat_remote_cusparse, _mat_remote->rows(), _mat_remote->cols(),
        _mat_remote->nonZeros(), _d_rowptr_remote, _d_colind_remote,
        _d_values_remote, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&_mat_remote_desc));
    CHECK_CUSPARSE(
        cusparseSetMatType(_mat_remote_desc, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(
        cusparseSetMatIndexBase(_mat_remote_desc, CUSPARSE_INDEX_BASE_ZERO));
  }
}
//-----------------------------------------------------------------------------
template <>
void Matrix<double>::cusparse_init()
{
  // Get handle to the CUSPARSE context
  CHECK_CUSPARSE(cusparseCreate(&_cusparse_handle));

  // Create cuSPARSE CSR wrapper for local block
  CHECK_CUSPARSE(cusparseCreateCsr(
      &_mat_local_cusparse, _mat_local->rows(), _mat_local->cols(),
      _mat_local->nonZeros(), _d_rowptr_local, _d_colind_local, _d_values_local,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_64F));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&_mat_local_desc));
  CHECK_CUSPARSE(
      cusparseSetMatType(_mat_local_desc, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE(
      cusparseSetMatIndexBase(_mat_local_desc, CUSPARSE_INDEX_BASE_ZERO));

  // Create cuSPARSE CSR wrapper for remote block if applicable
  if (_mat_remote != nullptr) {
    CHECK_CUSPARSE(cusparseCreateCsr(
        &_mat_remote_cusparse, _mat_remote->rows(), _mat_remote->cols(),
        _mat_remote->nonZeros(), _d_rowptr_remote, _d_colind_remote,
        _d_values_remote, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&_mat_remote_desc));
    CHECK_CUSPARSE(
        cusparseSetMatType(_mat_remote_desc, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(
        cusparseSetMatIndexBase(_mat_remote_desc, CUSPARSE_INDEX_BASE_ZERO));
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::cusparse_destroy()
{
  CHECK_CUSPARSE(cusparseDestroySpMat(_mat_local_cusparse));
  CHECK_CUSPARSE(cusparseDestroyMatDescr(_mat_local_desc));
  CHECK_CUSPARSE(cusparseDestroy(_cusparse_handle));
  CHECK_CUDA(cudaFree(_d_rowptr_local));
  CHECK_CUDA(cudaFree(_d_colind_local));
  CHECK_CUDA(cudaFree(_d_values_local));
  if (_buffer == nullptr)
    CHECK_CUDA(cudaFree(_buffer));
  if (_buffer_rmt == nullptr)
    CHECK_CUDA(cudaFree(_buffer_rmt));

  // FIXME
  if (_mat_remote != nullptr) {
    CHECK_CUSPARSE(cusparseDestroySpMat(_mat_remote_cusparse));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(_mat_remote_desc));
    CHECK_CUDA(cudaFree(_d_rowptr_remote));
    CHECK_CUDA(cudaFree(_d_colind_remote));
    CHECK_CUDA(cudaFree(_d_values_remote));
  }
}
//-----------------------------------------------------------------------------
template <>
void Matrix<float>::mult(float* x, float* y, cudaStream_t& stream) const
{
  CHECK_CUSPARSE(cusparseSetStream(_cusparse_handle, stream));

  cusparseDnVecDescr_t vec_x = nullptr;
  CHECK_CUSPARSE(cusparseCreateDnVec(
      &vec_x, _col_map->local_size() + _col_map->num_ghosts(), x, CUDA_R_32F));
  cusparseDnVecDescr_t vec_y = nullptr;
  CHECK_CUSPARSE(
      cusparseCreateDnVec(&vec_y, _mat_local->rows(), y, CUDA_R_32F));

  float alpha = 1.0;
  float beta = 0.0;

  // Allocate workspace for cuSPARSE
  if (_buffer == nullptr) {
    size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        _cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
        _mat_local_cusparse, vec_x, &beta, vec_y, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
    CHECK_CUDA(cudaMalloc(&_buffer, buffer_size));
  }

  // Allocate workspace for remote block if applicable
  if (_buffer_rmt == nullptr) {
    size_t buffer_size_remote = 0;
    if (_col_map->overlapping() && _mat_remote->nonZeros() > 0) {
      CHECK_CUSPARSE(cusparseSpMV_bufferSize(
          _cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          _mat_remote_cusparse, vec_x, &beta, vec_y, CUDA_R_32F,
          CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size_remote));
      CHECK_CUDA(cudaMalloc(&_buffer_rmt, buffer_size_remote));
    }
  }

  // Launch SpMV on local block
  CHECK_CUSPARSE(cusparseSpMV(_cusparse_handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              _mat_local_cusparse, vec_x, &beta, vec_y,
                              CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, _buffer));

  // Launch SpMV on remote block if applicable
  if (_col_map->overlapping() && _mat_remote->nonZeros() > 0) {
    _col_map->update_finalise(x, stream);
    beta = 1.0;
    CHECK_CUSPARSE(
        cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                     _mat_remote_cusparse, vec_x, &beta, vec_y, CUDA_R_32F,
                     CUSPARSE_SPMV_ALG_DEFAULT, _buffer_rmt));
  }

  CHECK_CUSPARSE(cusparseDestroyDnVec(vec_x));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vec_y));
}
//-----------------------------------------------------------------------------
template <>
void Matrix<double>::mult(double* x, double* y, cudaStream_t& stream) const
{
  CHECK_CUSPARSE(cusparseSetStream(_cusparse_handle, stream));

  cusparseDnVecDescr_t vec_x = nullptr;
  CHECK_CUSPARSE(cusparseCreateDnVec(
      &vec_x, _col_map->local_size() + _col_map->num_ghosts(), x, CUDA_R_64F));
  cusparseDnVecDescr_t vec_y = nullptr;
  CHECK_CUSPARSE(
      cusparseCreateDnVec(&vec_y, _mat_local->rows(), y, CUDA_R_64F));

  double alpha = 1.0;
  double beta = 0.0;

  // Allocate workspace for local block (only in the first call to SpMV)
  if (_buffer == nullptr) {
    size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        _cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
        _mat_local_cusparse, vec_x, &beta, vec_y, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
    CHECK_CUDA(cudaMalloc(&_buffer, buffer_size));
  }

  // Allocate workspace for remote block if applicable (only in the first call
  // to SpMV)
  if (_buffer_rmt == nullptr) {
    size_t buffer_size_remote = 0;
    if (_col_map->overlapping() && _mat_remote->nonZeros() > 0) {
      CHECK_CUSPARSE(cusparseSpMV_bufferSize(
          _cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          _mat_remote_cusparse, vec_x, &beta, vec_y, CUDA_R_64F,
          CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size_remote));
      CHECK_CUDA(cudaMalloc(&_buffer_rmt, buffer_size_remote));
    }
  }

  // Launch SpMV on local block
  CHECK_CUSPARSE(cusparseSpMV(_cusparse_handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              _mat_local_cusparse, vec_x, &beta, vec_y,
                              CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, _buffer));

  // Launch SpMV on remote block if applicable
  if (_col_map->overlapping() && _mat_remote->nonZeros() > 0) {
    _col_map->update_finalise(x, stream);
    beta = 1.0;
    CHECK_CUSPARSE(
        cusparseSpMV(_cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                     _mat_remote_cusparse, vec_x, &beta, vec_y, CUDA_R_64F,
                     CUSPARSE_SPMV_ALG_DEFAULT, _buffer_rmt));
  }

  CHECK_CUSPARSE(cusparseDestroyDnVec(vec_x));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vec_y));
}
//-----------------------------------------------------------------------------

// Explicit instantiation
template void spmv::Matrix<double>::cusparse_destroy();
template void spmv::Matrix<float>::cusparse_destroy();

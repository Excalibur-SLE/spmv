// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <cusparse.h>

#include "L2GMap.h"
#include "Matrix.h"
#include "helper_cuda.h"

namespace spmv
{

constexpr int BLOCK_SIZE = 512;

struct cusparse_data_t {
  cusparseHandle_t _handle = 0;
  cusparseSpMatDescr_t _mat_local = nullptr;
  cusparseSpMatDescr_t _mat_remote = nullptr;
  cusparseMatDescr_t _mat_local_desc = 0;
  cusparseMatDescr_t _mat_remote_desc = 0;
};

//-----------------------------------------------------------------------------
template <typename T>
__global__ void csr_sym_gmem_atomics(const int* __restrict__ rowptr,
                                     const int* __restrict__ colind,
                                     const T* __restrict__ values,
                                     const T* __restrict__ diagonal,
                                     const int nrows, const T* __restrict__ x,
                                     T* __restrict__ y)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nrows) {
    // Diagonal contribution
    T y_tmp = diagonal[i] * x[i];

    if (rowptr != nullptr) {
      // Compute symmetric SpMV on local block
      for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        int col = colind[j];
        T val = values[j];
        y_tmp += val * x[col];
        atomicAdd(&y[col], val * x[i]);
      }
    }

    atomicAdd(&y[i], y_tmp);
  }
}
//-----------------------------------------------------------------------------
template <typename T>
__global__ void csr_sym_shmem_atomics(const int* __restrict__ rowptr,
                                      const int* __restrict__ colind,
                                      const T* __restrict__ values,
                                      const T* __restrict__ diagonal,
                                      const int nrows, const T* __restrict__ x,
                                      T* __restrict__ y)
{
  __shared__ T _y[BLOCK_SIZE];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;
  const int block_offset = blockIdx.x * blockDim.x;
  T y_tmp = 0;
  T _x;

  if (i < nrows) {
    _x = x[i];
  }
  _y[tid] = 0;
  __syncthreads();

  if (i < nrows) {
    if (rowptr != nullptr) {
      // Compute symmetric SpMV on local block
      for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        int col = colind[j];
        T val = values[j];
        y_tmp += val * x[col];
        if (col < block_offset) {
          // If there is a conflict outside the team's range, update global
          // vector atomically
          atomicAdd(&y[col], val * _x);
        } else {
          // If there is a conflict within the team's range, update local vector
          // atomically
          atomicAdd_block(&_y[col - block_offset], val * _x);
        }
      }
    }
  }

  __syncthreads();
  if (i < nrows)
    atomicAdd(&y[i], diagonal[i] * _x + y_tmp + _y[tid]);
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::spmv_sym_cuda(const T* __restrict__ x, T* __restrict__ y,
                              cudaStream_t& stream) const
{
  const int block_size = BLOCK_SIZE;
  const int num_blocks = (rows() + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  csr_sym_gmem_atomics<<<dimGrid, dimBlock, 0, stream>>>(
      _d_rowptr_local, _d_colind_local, _d_values_local, _d_diagonal, rows(), x,
      y);
}
//-----------------------------------------------------------------------------
template <>
void Matrix<float>::cuda_init()
{
  _cusparse_data = new cusparse_data_t;

  // Get handle to the CUSPARSE context
  CHECK_CUSPARSE(cusparseCreate(&(_cusparse_data->_handle)));

  // Setup local part
  if (_mat_local != nullptr) {
    int nrows = _mat_local->rows();
    int nnz = _mat_local->nonZeros();
    if (nnz > 0) {
      CHECK_CUDA(
          cudaMalloc((void**)&_d_rowptr_local, (nrows + 1) * sizeof(int)));
      CHECK_CUDA(cudaMalloc((void**)&_d_colind_local, nnz * sizeof(int)));
      CHECK_CUDA(cudaMalloc((void**)&_d_values_local, nnz * sizeof(float)));
      CHECK_CUDA(cudaMemcpy(_d_rowptr_local, _mat_local->outerIndexPtr(),
                            (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(_d_colind_local, _mat_local->innerIndexPtr(),
                            nnz * sizeof(int), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(_d_values_local, _mat_local->valuePtr(),
                            nnz * sizeof(float), cudaMemcpyHostToDevice));

      // Create cuSPARSE CSR wrapper for local block
      CHECK_CUSPARSE(cusparseCreateCsr(
          &(_cusparse_data->_mat_local), _mat_local->rows(), _mat_local->cols(),
          _mat_local->nonZeros(), _d_rowptr_local, _d_colind_local,
          _d_values_local, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
      CHECK_CUSPARSE(
          cusparseCreateMatDescr(&(_cusparse_data->_mat_local_desc)));
      CHECK_CUSPARSE(cusparseSetMatType(_cusparse_data->_mat_local_desc,
                                        CUSPARSE_MATRIX_TYPE_GENERAL));
      CHECK_CUSPARSE(cusparseSetMatIndexBase(_cusparse_data->_mat_local_desc,
                                             CUSPARSE_INDEX_BASE_ZERO));
    }
  }

  // Setup remote part
  if (_mat_remote != nullptr) {
    int nrows = _mat_remote->rows();
    int nnz = _mat_remote->nonZeros();
    if (nnz > 0) {
      int ncols_padded = _mat_remote->cols();
      CHECK_CUDA(
          cudaMalloc((void**)&_d_rowptr_remote, (nrows + 1) * sizeof(int)));
      CHECK_CUDA(cudaMalloc((void**)&_d_rowind_remote, nnz * sizeof(int)));
      CHECK_CUDA(cudaMalloc((void**)&_d_colind_remote, nnz * sizeof(int)));
      CHECK_CUDA(cudaMalloc((void**)&_d_values_remote, nnz * sizeof(float)));
      CHECK_CUDA(cudaMemcpy(_d_rowptr_remote, _mat_remote->outerIndexPtr(),
                            (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(_d_colind_remote, _mat_remote->innerIndexPtr(),
                            nnz * sizeof(int), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(_d_values_remote, _mat_remote->valuePtr(),
                            nnz * sizeof(float), cudaMemcpyHostToDevice));

      // Create cuSPARSE COO wrapper for remote block if applicable
      CHECK_CUSPARSE(cusparseXcsr2coo(_cusparse_data->_handle, _d_rowptr_remote,
                                      nnz, nrows, _d_rowind_remote,
                                      CUSPARSE_INDEX_BASE_ZERO));
      CHECK_CUSPARSE(cusparseCreateCoo(
          &(_cusparse_data->_mat_remote), nrows, ncols_padded, nnz,
          _d_rowind_remote, _d_colind_remote, _d_values_remote,
          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
      CHECK_CUSPARSE(
          cusparseCreateMatDescr(&(_cusparse_data->_mat_remote_desc)));
      CHECK_CUSPARSE(cusparseSetMatType(_cusparse_data->_mat_remote_desc,
                                        CUSPARSE_MATRIX_TYPE_GENERAL));
      CHECK_CUSPARSE(cusparseSetMatIndexBase(_cusparse_data->_mat_remote_desc,
                                             CUSPARSE_INDEX_BASE_ZERO));
      CHECK_CUDA(cudaFree(_d_rowptr_remote));
    }
  }

  // Setup diagonal part, if applicable
  if (_symmetric) {
    CHECK_CUDA(cudaMalloc((void**)&_d_diagonal, rows() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(_d_diagonal, _mat_diagonal->data(),
                          rows() * sizeof(float), cudaMemcpyHostToDevice));
  }
}
//-----------------------------------------------------------------------------
template <>
void Matrix<double>::cuda_init()
{
  _cusparse_data = new cusparse_data_t;

  // Get handle to the CUSPARSE context
  CHECK_CUSPARSE(cusparseCreate(&(_cusparse_data->_handle)));

  // Setup local part
  if (_mat_local != nullptr) {
    int nrows = _mat_local->rows();
    int nnz = _mat_local->nonZeros();
    if (nnz > 0) {
      CHECK_CUDA(
          cudaMalloc((void**)&_d_rowptr_local, (nrows + 1) * sizeof(int)));
      CHECK_CUDA(cudaMalloc((void**)&_d_colind_local, nnz * sizeof(int)));
      CHECK_CUDA(cudaMalloc((void**)&_d_values_local, nnz * sizeof(double)));
      CHECK_CUDA(cudaMemcpy(_d_rowptr_local, _mat_local->outerIndexPtr(),
                            (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(_d_colind_local, _mat_local->innerIndexPtr(),
                            nnz * sizeof(int), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(_d_values_local, _mat_local->valuePtr(),
                            nnz * sizeof(double), cudaMemcpyHostToDevice));

      // Create cuSPARSE CSR wrapper for local block
      CHECK_CUSPARSE(cusparseCreateCsr(
          &(_cusparse_data->_mat_local), _mat_local->rows(), _mat_local->cols(),
          _mat_local->nonZeros(), _d_rowptr_local, _d_colind_local,
          _d_values_local, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
      CHECK_CUSPARSE(
          cusparseCreateMatDescr(&(_cusparse_data->_mat_local_desc)));
      CHECK_CUSPARSE(cusparseSetMatType(_cusparse_data->_mat_local_desc,
                                        CUSPARSE_MATRIX_TYPE_GENERAL));
      CHECK_CUSPARSE(cusparseSetMatIndexBase(_cusparse_data->_mat_local_desc,
                                             CUSPARSE_INDEX_BASE_ZERO));
    }
  }

  // Setup remote part
  if (_mat_remote != nullptr) {
    int nrows = _mat_remote->rows();
    int nnz = _mat_remote->nonZeros();
    if (nnz > 0) {
      int ncols_padded = _mat_remote->cols();
      CHECK_CUDA(
          cudaMalloc((void**)&_d_rowptr_remote, (nrows + 1) * sizeof(int)));
      CHECK_CUDA(cudaMalloc((void**)&_d_rowind_remote, nnz * sizeof(int)));
      CHECK_CUDA(cudaMalloc((void**)&_d_colind_remote, nnz * sizeof(int)));
      CHECK_CUDA(cudaMalloc((void**)&_d_values_remote, nnz * sizeof(double)));
      CHECK_CUDA(cudaMemcpy(_d_rowptr_remote, _mat_remote->outerIndexPtr(),
                            (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(_d_colind_remote, _mat_remote->innerIndexPtr(),
                            nnz * sizeof(int), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(_d_values_remote, _mat_remote->valuePtr(),
                            nnz * sizeof(double), cudaMemcpyHostToDevice));

      // Create cuSPARSE COO wrapper for remote block if applicable
      CHECK_CUSPARSE(cusparseXcsr2coo(_cusparse_data->_handle, _d_rowptr_remote,
                                      nnz, nrows, _d_rowind_remote,
                                      CUSPARSE_INDEX_BASE_ZERO));
      CHECK_CUSPARSE(cusparseCreateCoo(
          &(_cusparse_data->_mat_remote), nrows, ncols_padded, nnz,
          _d_rowind_remote, _d_colind_remote, _d_values_remote,
          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
      CHECK_CUSPARSE(
          cusparseCreateMatDescr(&(_cusparse_data->_mat_remote_desc)));
      CHECK_CUSPARSE(cusparseSetMatType(_cusparse_data->_mat_remote_desc,
                                        CUSPARSE_MATRIX_TYPE_GENERAL));
      CHECK_CUSPARSE(cusparseSetMatIndexBase(_cusparse_data->_mat_remote_desc,
                                             CUSPARSE_INDEX_BASE_ZERO));
      CHECK_CUDA(cudaFree(_d_rowptr_remote));
    }
  }

  // Setup diagonal part, if applicable
  if (_symmetric) {
    CHECK_CUDA(cudaMalloc((void**)&_d_diagonal, rows() * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(_d_diagonal, _mat_diagonal->data(),
                          rows() * sizeof(double), cudaMemcpyHostToDevice));
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::cuda_destroy()
{
  CHECK_CUDA(cudaFree(_d_rowptr_local));
  CHECK_CUDA(cudaFree(_d_colind_local));
  CHECK_CUDA(cudaFree(_d_values_local));
  CHECK_CUDA(cudaFree(_d_diagonal));
  CHECK_CUDA(cudaFree(_d_rowind_remote));
  CHECK_CUDA(cudaFree(_d_colind_remote));
  CHECK_CUDA(cudaFree(_d_values_remote));
  CHECK_CUSPARSE(cusparseDestroySpMat(_cusparse_data->_mat_local));
  CHECK_CUSPARSE(cusparseDestroyMatDescr(_cusparse_data->_mat_local_desc));
  CHECK_CUSPARSE(cusparseDestroySpMat(_cusparse_data->_mat_remote));
  CHECK_CUSPARSE(cusparseDestroyMatDescr(_cusparse_data->_mat_remote_desc));
  CHECK_CUSPARSE(cusparseDestroy(_cusparse_data->_handle));
  CHECK_CUDA(cudaFree(_buffer));
  CHECK_CUDA(cudaFree(_buffer_rmt));
}
//-----------------------------------------------------------------------------
template <>
void Matrix<float>::mult(float* __restrict__ x, float* __restrict__ y,
                         cudaStream_t& stream) const
{
  assert(x != nullptr && y != nullptr);

  if (_symmetric) {
    // Zero-out output vector
    CHECK_CUDA(cudaMemsetAsync(y, 0, rows() * sizeof(float), stream));

    if (_mat_local != nullptr) {
      if (_mat_local->nonZeros() > 0) {
        spmv_sym_cuda(x, y, stream);
      }
    }

    // Launch in a different stream to hide transfer
    cudaStream_t transfer_stream;
    cudaEvent_t transfer_event;
    CHECK_CUDA(
        cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking));
    if (_col_map->overlapping()) {
      _col_map->update_finalise(x, transfer_stream);
    }
    CHECK_CUDA(
        cudaEventCreateWithFlags(&transfer_event, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventRecord(transfer_event, transfer_stream));

    // Launch SpMV on remote block if applicable
    if (_mat_remote != nullptr) {
      if (_mat_remote->nonZeros() > 0) {
        double alpha = 1.0;
        double beta = 1.0;
        cusparseDnVecDescr_t vec_x = nullptr;
        CHECK_CUSPARSE(
            cusparseCreateDnVec(&vec_x, _mat_remote->cols(), x, CUDA_R_32F));
        cusparseDnVecDescr_t vec_y = nullptr;
        CHECK_CUSPARSE(
            cusparseCreateDnVec(&vec_y, _mat_remote->rows(), y, CUDA_R_32F));

        // Allocate workspace for remote block if applicable
        if (_buffer_rmt == nullptr) {
          size_t buffer_size_remote = 0;
          CHECK_CUSPARSE(cusparseSpMV_bufferSize(
              _cusparse_data->_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
              _cusparse_data->_mat_remote, vec_x, &beta, vec_y, CUDA_R_32F,
              CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size_remote));
          CHECK_CUDA(cudaMalloc(&_buffer_rmt, buffer_size_remote));
        }
        CHECK_CUDA(cudaStreamWaitEvent(stream, transfer_event));
        CHECK_CUSPARSE(cusparseSpMV(
            _cusparse_data->_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            _cusparse_data->_mat_remote, vec_x, &beta, vec_y, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, _buffer_rmt));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vec_x));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vec_y));
      }
    }
    CHECK_CUDA(cudaStreamDestroy(transfer_stream));
    CHECK_CUDA(cudaEventDestroy(transfer_event));
  } else {
    CHECK_CUSPARSE(cusparseSetStream(_cusparse_data->_handle, stream));

    cusparseDnVecDescr_t vec_x = nullptr;
    CHECK_CUSPARSE(
        cusparseCreateDnVec(&vec_x, _mat_local->cols(), x, CUDA_R_32F));
    cusparseDnVecDescr_t vec_y = nullptr;
    CHECK_CUSPARSE(
        cusparseCreateDnVec(&vec_y, _mat_local->rows(), y, CUDA_R_32F));

    float alpha = 1.0;
    float beta = 0.0;

    // Allocate workspace for cuSPARSE
    if (_buffer == nullptr) {
      size_t buffer_size = 0;
      CHECK_CUSPARSE(cusparseSpMV_bufferSize(
          _cusparse_data->_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          _cusparse_data->_mat_local, vec_x, &beta, vec_y, CUDA_R_32F,
          CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
      CHECK_CUDA(cudaMallocAsync(&_buffer, buffer_size, stream));
    }

    // Launch SpMV on local block
    CHECK_CUSPARSE(
        cusparseSpMV(_cusparse_data->_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, _cusparse_data->_mat_local, vec_x, &beta, vec_y,
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, _buffer));

    // Launch in a different stream to hide transfer
    cudaStream_t transfer_stream;
    cudaEvent_t transfer_event;
    CHECK_CUDA(
        cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking));
    if (_col_map->overlapping()) {
      _col_map->update_finalise(x, transfer_stream);
    }
    CHECK_CUDA(
        cudaEventCreateWithFlags(&transfer_event, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventRecord(transfer_event, transfer_stream));

    // Launch SpMV on remote block if applicable
    if (_mat_remote != nullptr) {
      if (_mat_remote->nonZeros() > 0) {
        cusparseDnVecDescr_t vec_x = nullptr;
        CHECK_CUSPARSE(
            cusparseCreateDnVec(&vec_x, _mat_remote->cols(), x, CUDA_R_32F));
        beta = 1.0;
        // Allocate workspace for remote block if applicable
        if (_buffer_rmt == nullptr) {
          size_t buffer_size_remote = 0;
          CHECK_CUSPARSE(cusparseSpMV_bufferSize(
              _cusparse_data->_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
              _cusparse_data->_mat_remote, vec_x, &beta, vec_y, CUDA_R_32F,
              CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size_remote));
          CHECK_CUDA(cudaMallocAsync(&_buffer_rmt, buffer_size_remote, stream));
        }
        CHECK_CUDA(cudaStreamWaitEvent(stream, transfer_event));
        CHECK_CUSPARSE(cusparseSpMV(
            _cusparse_data->_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            _cusparse_data->_mat_remote, vec_x, &beta, vec_y, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, _buffer_rmt));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vec_x));
      }
    }

    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_x));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_y));
    CHECK_CUDA(cudaStreamDestroy(transfer_stream));
    CHECK_CUDA(cudaEventDestroy(transfer_event));
  }
}
//-----------------------------------------------------------------------------
template <>
void Matrix<double>::mult(double* __restrict__ x, double* __restrict__ y,
                          cudaStream_t& stream) const
{
  assert(x != nullptr && y != nullptr);

  if (_symmetric) {
    // Zero-out output vector
    CHECK_CUDA(cudaMemsetAsync(y, 0, rows() * sizeof(double), stream));

    spmv_sym_cuda(x, y, stream);

    // Launch in a different stream to hide transfer
    cudaStream_t transfer_stream;
    cudaEvent_t transfer_event;
    CHECK_CUDA(
        cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking));
    if (_col_map->overlapping()) {
      _col_map->update_finalise(x, transfer_stream);
    }
    CHECK_CUDA(
        cudaEventCreateWithFlags(&transfer_event, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventRecord(transfer_event, transfer_stream));

    // Launch SpMV on remote block if applicable
    if (_mat_remote != nullptr) {
      if (_mat_remote->nonZeros() > 0) {
        double alpha = 1.0;
        double beta = 1.0;
        cusparseDnVecDescr_t vec_x = nullptr;
        CHECK_CUSPARSE(
            cusparseCreateDnVec(&vec_x, _mat_remote->cols(), x, CUDA_R_64F));
        cusparseDnVecDescr_t vec_y = nullptr;
        CHECK_CUSPARSE(
            cusparseCreateDnVec(&vec_y, _mat_remote->rows(), y, CUDA_R_64F));

        // Allocate workspace for remote block if applicable
        if (_buffer_rmt == nullptr) {
          size_t buffer_size_remote = 0;
          CHECK_CUSPARSE(cusparseSpMV_bufferSize(
              _cusparse_data->_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
              _cusparse_data->_mat_remote, vec_x, &beta, vec_y, CUDA_R_64F,
              CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size_remote));
          CHECK_CUDA(cudaMallocAsync(&_buffer_rmt, buffer_size_remote, stream));
        }
        CHECK_CUDA(cudaStreamWaitEvent(stream, transfer_event));
        CHECK_CUSPARSE(cusparseSpMV(
            _cusparse_data->_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            _cusparse_data->_mat_remote, vec_x, &beta, vec_y, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, _buffer_rmt));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vec_x));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vec_y));
      }
    }
    CHECK_CUDA(cudaStreamDestroy(transfer_stream));
    CHECK_CUDA(cudaEventDestroy(transfer_event));
  } else {
    CHECK_CUSPARSE(cusparseSetStream(_cusparse_data->_handle, stream));
    double alpha = 1.0;
    double beta = 0.0;
    cusparseDnVecDescr_t vec_x = nullptr;
    CHECK_CUSPARSE(
        cusparseCreateDnVec(&vec_x, _mat_local->cols(), x, CUDA_R_64F));
    cusparseDnVecDescr_t vec_y = nullptr;
    CHECK_CUSPARSE(
        cusparseCreateDnVec(&vec_y, _mat_local->rows(), y, CUDA_R_64F));

    // Allocate workspace for local block (only in the first call to SpMV)
    if (_buffer == nullptr) {
      size_t buffer_size = 0;
      CHECK_CUSPARSE(cusparseSpMV_bufferSize(
          _cusparse_data->_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          _cusparse_data->_mat_local, vec_x, &beta, vec_y, CUDA_R_64F,
          CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
      CHECK_CUDA(cudaMallocAsync(&_buffer, buffer_size, stream));
    }

    // Launch SpMV on local block
    CHECK_CUSPARSE(
        cusparseSpMV(_cusparse_data->_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, _cusparse_data->_mat_local, vec_x, &beta, vec_y,
                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, _buffer));

    // Launch in a different stream to hide transfer
    cudaStream_t transfer_stream;
    cudaEvent_t transfer_event;
    CHECK_CUDA(
        cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking));
    if (_col_map->overlapping()) {
      _col_map->update_finalise(x, transfer_stream);
    }
    CHECK_CUDA(
        cudaEventCreateWithFlags(&transfer_event, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventRecord(transfer_event, transfer_stream));

    // Launch SpMV on remote block if applicable
    if (_mat_remote != nullptr) {
      if (_mat_remote->nonZeros() > 0) {
        cusparseDnVecDescr_t vec_x = nullptr;
        CHECK_CUSPARSE(
            cusparseCreateDnVec(&vec_x, _mat_remote->cols(), x, CUDA_R_64F));
        beta = 1.0;
        // Allocate workspace for remote block (only in the first call to SpMV)
        if (_buffer_rmt == nullptr) {
          size_t buffer_size_remote = 0;
          CHECK_CUSPARSE(cusparseSpMV_bufferSize(
              _cusparse_data->_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
              _cusparse_data->_mat_remote, vec_x, &beta, vec_y, CUDA_R_64F,
              CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size_remote));
          CHECK_CUDA(cudaMallocAsync(&_buffer_rmt, buffer_size_remote, stream));
        }
        CHECK_CUDA(cudaStreamWaitEvent(stream, transfer_event));
        CHECK_CUSPARSE(cusparseSpMV(
            _cusparse_data->_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            _cusparse_data->_mat_remote, vec_x, &beta, vec_y, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, _buffer_rmt));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vec_x));
      }
    }

    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_x));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_y));
    CHECK_CUDA(cudaStreamDestroy(transfer_stream));
    CHECK_CUDA(cudaEventDestroy(transfer_event));
  }
}
//-----------------------------------------------------------------------------

// Explicit instantiation
template void spmv::Matrix<double>::cuda_destroy();
template void spmv::Matrix<float>::cuda_destroy();

} // namespace spmv

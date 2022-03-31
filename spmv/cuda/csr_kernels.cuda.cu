// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "csr_kernels.h"
#include "cuda_executor.h"

namespace spmv
{

struct cusparse_data_t {
  cusparseSpMatDescr_t _mat = nullptr;
  cusparseMatDescr_t _mat_desc = nullptr;
  char* _buffer = nullptr;
};

template <typename T>
struct static_false : std::false_type {
};

constexpr int BLOCK_SIZE = 512;

template <typename T>
__global__ void csr_sym_gmem_atomics(const int32_t* __restrict__ rowptr,
                                     const int32_t* __restrict__ colind,
                                     const T* __restrict__ values,
                                     const T* __restrict__ diagonal,
                                     const int num_rows, const T alpha,
                                     const T* __restrict__ in,
                                     T* __restrict__ out)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < num_rows) {
    T sum = diagonal[i] * in[i];

    if (rowptr != nullptr) {
      for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        int col = colind[j];
        T val = values[j];
        sum += val * in[col];
        atomicAdd(&out[col], alpha * val * in[i]);
      }
    }

    atomicAdd(&out[i], alpha * sum);
  }
}

template <typename T>
void CSRSpMV<T>::init(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
                      int32_t* rowptr, int32_t* colind, T* values,
                      bool symmetric, const CudaExecutor& exec)
{
  _symmetric = symmetric;
  // FIXME use host exec
  if (!_symmetric && num_non_zeros > 0) {
    _aux_data = new cusparse_data_t;
    cusparse_data_t* cusparse_data = (cusparse_data_t*)_aux_data;
    if constexpr (std::is_same<T, float>()) {
      CHECK_CUSPARSE(cusparseCreateCsr(
          &(cusparse_data->_mat), num_rows, num_cols, num_non_zeros, rowptr,
          colind, values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    } else if constexpr (std::is_same<T, double>()) {
      CHECK_CUSPARSE(cusparseCreateCsr(
          &(cusparse_data->_mat), num_rows, num_cols, num_non_zeros, rowptr,
          colind, values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    } else {
      static_assert(static_false<T>::value);
    }
    CHECK_CUSPARSE(cusparseCreateMatDescr(&(cusparse_data->_mat_desc)));
    CHECK_CUSPARSE(cusparseSetMatType(cusparse_data->_mat_desc,
                                      CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(cusparse_data->_mat_desc,
                                           CUSPARSE_INDEX_BASE_ZERO));
  }
}

template <typename T>
void CSRSpMV<T>::run(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
                     const int32_t* rowptr, const int32_t* colind,
                     const T* values, const T* diagonal, T alpha,
                     T* __restrict__ in, T beta, T* __restrict__ out,
                     const CudaExecutor& exec) const
{
  if (_symmetric) {
    // Scale by beta
    if constexpr (std::is_same<T, float>()) {
      CHECK_CUBLAS(
          cublasSscal(exec.get_cublas_handle(), num_rows, &beta, out, 1));
    } else if constexpr (std::is_same<T, double>()) {
      CHECK_CUBLAS(
          cublasDscal(exec.get_cublas_handle(), num_rows, &beta, out, 1));
    } else {
      static_assert(static_false<T>::value);
    }

    const int block_size = BLOCK_SIZE;
    const int num_blocks = (num_rows + block_size - 1) / block_size;
    dim3 dimBlock(block_size);
    dim3 dimGrid(num_blocks);
    csr_sym_gmem_atomics<<<dimGrid, dimBlock, 0, exec.get_cuda_stream()>>>(
        rowptr, colind, values, diagonal, num_rows, alpha, in, out);
    exec.synchronize();
  } else {
    cusparse_data_t* cusparse_data = (cusparse_data_t*)_aux_data;
    cusparseDnVecDescr_t vec_in = nullptr;
    cusparseDnVecDescr_t vec_out = nullptr;
    if constexpr (std::is_same<T, float>()) {
      CHECK_CUSPARSE(cusparseCreateDnVec(&vec_in, num_cols, in, CUDA_R_32F));
      CHECK_CUSPARSE(cusparseCreateDnVec(&vec_out, num_rows, out, CUDA_R_32F));

      // Allocate workspace for cuSPARSE
      if (cusparse_data->_buffer == nullptr) {
        size_t buffer_size = 0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            exec.get_cusparse_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, cusparse_data->_mat, vec_in, &beta, vec_out, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
        cusparse_data->_buffer = exec.alloc<char>(buffer_size);
      }

      // Launch SpMV
      CHECK_CUSPARSE(cusparseSpMV(
          exec.get_cusparse_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          cusparse_data->_mat, vec_in, &beta, vec_out, CUDA_R_32F,
          CUSPARSE_SPMV_ALG_DEFAULT, cusparse_data->_buffer));
    } else if constexpr (std::is_same<T, double>()) {
      CHECK_CUSPARSE(cusparseCreateDnVec(&vec_in, num_cols, in, CUDA_R_64F));
      CHECK_CUSPARSE(cusparseCreateDnVec(&vec_out, num_rows, out, CUDA_R_64F));

      // Allocate workspace for cuSPARSE
      if (cusparse_data->_buffer == nullptr) {
        size_t buffer_size = 0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            exec.get_cusparse_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, cusparse_data->_mat, vec_in, &beta, vec_out, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
        cusparse_data->_buffer = exec.alloc<char>(buffer_size);
      }

      // Launch SpMV
      CHECK_CUSPARSE(cusparseSpMV(
          exec.get_cusparse_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          cusparse_data->_mat, vec_in, &beta, vec_out, CUDA_R_64F,
          CUSPARSE_SPMV_ALG_DEFAULT, cusparse_data->_buffer));
    } else {
      static_assert(static_false<T>::value);
    }
  }
}

template <typename T>
void CSRSpMV<T>::finalize(const CudaExecutor& exec) const
{
  if (_aux_data != nullptr) {
    cusparse_data_t* cusparse_data = (cusparse_data_t*)_aux_data;
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_data->_mat));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(cusparse_data->_mat_desc));
    delete cusparse_data;
  }
}

} // namespace spmv

// Explicit instantiations
template class spmv::CSRSpMV<float>;
template class spmv::CSRSpMV<double>;

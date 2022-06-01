// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "coo_kernels.h"
#include "cuda_executor.h"
#include "cuda_helper.h"

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

template <typename T>
void COOSpMV<T>::init(int32_t num_rows, int32_t num_cols, int64_t num_non_zeros,
                      const int32_t* rowind, const int32_t* colind,
                      const T* values, const CudaExecutor& exec)
{
  _aux_data = malloc(sizeof(cusparse_data_t*));
  cusparse_data_t* cusparse_data = (cusparse_data_t*)_aux_data;
  if constexpr (std::is_same<T, float>()) {
    CHECK_CUSPARSE(cusparseCreateCoo(
        &(cusparse_data->_mat), num_rows, num_cols, num_non_zeros,
        (void*)rowind, (void*)colind, (void*)values, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  } else if constexpr (std::is_same<T, double>()) {
    CHECK_CUSPARSE(cusparseCreateCoo(
        &(cusparse_data->_mat), num_rows, num_cols, num_non_zeros,
        (void*)rowind, (void*)colind, (void*)values, CUSPARSE_INDEX_32I,
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

template <typename T>
void COOSpMV<T>::run(int32_t num_rows, int32_t num_cols, int64_t num_non_zeros,
                     const int32_t* rowind, const int32_t* colind,
                     const T* values, T alpha, T* __restrict__ in, T beta,
                     T* __restrict__ out, const CudaExecutor& exec) const
{
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
          exec.get_cusparse_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          cusparse_data->_mat, vec_in, &beta, vec_out, CUDA_R_32F,
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
          exec.get_cusparse_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          cusparse_data->_mat, vec_in, &beta, vec_out, CUDA_R_64F,
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

template <typename T>
void COOSpMV<T>::finalize(const CudaExecutor& exec) const
{
  cusparse_data_t* cusparse_data = (cusparse_data_t*)_aux_data;
  CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_data->_mat));
  CHECK_CUSPARSE(cusparseDestroyMatDescr(cusparse_data->_mat_desc));
}

} // namespace spmv

// Explicit instantiations
template class spmv::COOSpMV<float>;
template class spmv::COOSpMV<double>;

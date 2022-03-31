// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "coo_matrix.h"
#include "device_executor.h"

namespace spmv
{

template <typename T>
COOMatrix<T>::COOMatrix(const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat,
                        std::shared_ptr<DeviceExecutor> exec)
    : COOMatrix(mat.rows(), mat.cols(), mat.nonZeros(), mat.outerIndexPtr(),
                mat.innerIndexPtr(), mat.valuePtr(), exec)
{
}

template <typename T>
COOMatrix<T>::COOMatrix(int32_t num_rows, int32_t num_cols,
                        int32_t num_non_zeros, const int32_t* rowptr,
                        const int32_t* colind, const T* values,
                        std::shared_ptr<DeviceExecutor> exec)
{
  this->_exec = exec;
  this->_num_rows = num_rows;
  this->_num_cols = num_cols;
  this->_num_non_zeros = num_non_zeros;

  // If matrix is not empty
  if (num_non_zeros > 0) {
    // Convert rowptr to rowind
    std::vector<int32_t> rowind(num_non_zeros);
    int32_t cnt = 0;
    for (int i = 0; i < num_rows; i++) {
      for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
        rowind[cnt++] = i;
      }
    }
    // Allocate device buffers
    _rowind = exec->alloc<int32_t>(this->_num_non_zeros);
    _colind = exec->alloc<int32_t>(this->_num_non_zeros);
    _values = exec->alloc<T>(this->_num_non_zeros);
    // Copy data from Eigen Matrix using device executor
    // We assume Eigen data is using the reference executor
    exec->copy_from<int32_t>(_rowind, this->_exec->get_host(), rowind.data(),
                             this->_num_non_zeros);
    exec->copy_from<int32_t>(_colind, this->_exec->get_host(), colind,
                             this->_num_non_zeros);
    exec->copy_from<T>(_values, this->_exec->get_host(), values,
                       this->_num_non_zeros);
    // Initialize SpMV algorithm
    this->_exec->spmv_init(_op, *this);
  }
}

template <typename T>
COOMatrix<T>::~COOMatrix()
{
  // Finalize SpMV algorithm
  this->_exec->spmv_finalize(_op);
  this->_exec->free(_rowind);
  this->_exec->free(_colind);
  this->_exec->free(_values);
  this->_exec->free(this->_diagonal);
}

template <typename T>
size_t COOMatrix<T>::format_size() const
{
  size_t total_bytes;
  total_bytes = this->_num_non_zeros * (2 * sizeof(int) + sizeof(T));
  return total_bytes;
}

template <typename T>
void COOMatrix<T>::mult(T alpha, T* __restrict__ in, T beta,
                        T* __restrict__ out) const
{
  this->_exec->spmv_run(_op, *this, alpha, in, beta, out);
}

void _rowptr2rowind(const ReferenceExecutor& exec, const int32_t* rowptr,
                    int32_t* rowind)
{
  // FIXME
}

} // namespace spmv

// Explicit instantiations
template class spmv::COOMatrix<float>;
template class spmv::COOMatrix<double>;

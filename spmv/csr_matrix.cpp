// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "csr_matrix.h"
#include "device_executor.h"

namespace spmv
{

//-----------------------------------------------------------------------------
template <typename T>
CSRMatrix<T>::CSRMatrix(std::shared_ptr<DeviceExecutor> exec,
                        const Eigen::SparseMatrix<T, Eigen::RowMajor>* mat,
                        const Eigen::Matrix<T, Eigen::Dynamic, 1>* diagonal,
                        bool symmetric)
    : CSRMatrix(exec, mat->rows(), mat->cols(), mat->nonZeros(),
                mat->outerIndexPtr(), mat->innerIndexPtr(), mat->valuePtr(),
                diagonal ? diagonal->data() : nullptr, symmetric)
{
}
//-----------------------------------------------------------------------------
template <typename T>
CSRMatrix<T>::CSRMatrix(std::shared_ptr<DeviceExecutor> exec, int32_t num_rows,
                        int32_t num_cols, int64_t num_non_zeros,
                        const int32_t* rowptr, const int32_t* colind,
                        const T* values, const T* diagonal, bool symmetric)
{
  this->_exec = exec;
  this->_num_rows = num_rows;
  this->_num_cols = num_cols;
  this->_num_non_zeros = num_non_zeros;
  this->_symmetric = symmetric;

  // If matrix is not empty
  if (num_non_zeros > 0) {
    // Allocate device buffers
    _rowptr = exec->alloc<int32_t>(this->_num_rows + 1);
    _colind = exec->alloc<int32_t>(this->_num_non_zeros);
    _values = exec->alloc<T>(this->_num_non_zeros);
    // Copy data from Eigen Matrix using device executor
    // We assume Eigen data is using the reference executor
    exec->copy_from<int32_t>(_rowptr, this->_exec->get_host(), rowptr,
                             this->_num_rows + 1);
    exec->copy_from<int32_t>(_colind, this->_exec->get_host(), colind,
                             this->_num_non_zeros);
    exec->copy_from<T>(_values, this->_exec->get_host(), values,
                       this->_num_non_zeros);
  }

  // If diagonal is not empty
  if (diagonal != nullptr) {
    this->_diagonal = exec->alloc<T>(this->_num_rows);
    exec->copy_from<T>(this->_diagonal, this->_exec->get_host(), diagonal,
                       this->_num_rows);
  }

  // Initialize SpMV algorithm
  this->_exec->spmv_init(_op, *this);
}
//-----------------------------------------------------------------------------
template <typename T>
CSRMatrix<T>::~CSRMatrix()
{
  // Finalize SpMV algorithm
  this->_exec->spmv_finalize(_op);
  this->_exec->free(_rowptr);
  this->_exec->free(_colind);
  this->_exec->free(_values);
  this->_exec->free(this->_diagonal);
}
//-----------------------------------------------------------------------------
template <typename T>
size_t CSRMatrix<T>::format_size() const
{
  size_t total_bytes;
  total_bytes = (this->_num_rows + 1) * sizeof(int)
                + this->_num_non_zeros * (sizeof(int) + sizeof(T));
  return total_bytes;
}
//-----------------------------------------------------------------------------
template <typename T>
void CSRMatrix<T>::mult(T alpha, T* __restrict__ in, T beta,
                        T* __restrict__ out) const
{
  if (this->_num_non_zeros > 0 || this->_diagonal != nullptr)
    this->_exec->spmv_run(_op, *this, alpha, in, beta, out);
}
//-----------------------------------------------------------------------------

} // namespace spmv

// Explicit instantiations
template class spmv::CSRMatrix<float>;
template class spmv::CSRMatrix<double>;

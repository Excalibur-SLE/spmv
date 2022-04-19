// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "omp_offload_executor.h"
#include <cstring>
#include <iostream>
#include <typeinfo>

#include "coo_matrix.h"
#include "csr_matrix.h"

namespace spmv
{

OmpOffloadExecutor::OmpOffloadExecutor()
{
  this->_dev_info.device_id = omp_get_default_device();
}

OmpOffloadExecutor::OmpOffloadExecutor(int device_id)
{
  omp_set_default_device(device_id);
  this->_dev_info.device_id = device_id;
}

void OmpOffloadExecutor::synchronize() const {}

const DeviceExecutor& OmpOffloadExecutor::get_host() const { return *this; }

int OmpOffloadExecutor::get_num_devices() const
{
  // Returns the number of non-host devices available for offload
  return omp_get_num_devices();
}

int OmpOffloadExecutor::get_num_cus() const { return omp_get_num_devices(); }

void* OmpOffloadExecutor::_alloc(size_t num_bytes) const
{
  void* ptr = nullptr;
  ptr = omp_target_alloc(num_bytes, this->_dev_info.device_id);
  if (ptr == nullptr) {
    std::cerr << "ERROR: allocation on device failed" << std::endl;
    exit(1);
  }

  return ptr;
}

void OmpOffloadExecutor::_free(void* ptr) const
{
  omp_target_free(ptr, this->_dev_info.device_id);
}

void OmpOffloadExecutor::_memset(void* ptr, int value,
                                 size_t num_bytes) const {};

void OmpOffloadExecutor::_copy(void* dst_ptr, const void* src_ptr,
                               size_t num_bytes) const
{
  omp_target_memcpy(dst_ptr, src_ptr, num_bytes, 0, 0,
                    this->_dev_info.device_id, this->_dev_info.device_id);
}

void OmpOffloadExecutor::_copy_async(void* dst_ptr, const void* src_ptr,
                                     size_t num_bytes, void* obj) const
{
  _copy(dst_ptr, src_ptr, num_bytes);
}

void OmpOffloadExecutor::_copy_from(void* dst_ptr,
                                    const DeviceExecutor& src_exec,
                                    const void* src_ptr, size_t num_bytes) const
{
  omp_target_memcpy(dst_ptr, src_ptr, num_bytes, 0, 0,
                    this->_dev_info.device_id, omp_get_initial_device());
}

void OmpOffloadExecutor::_copy_to(void* dst_ptr, const DeviceExecutor& dst_exec,
                                  const void* src_ptr, size_t num_bytes) const
{
  omp_target_memcpy(dst_ptr, src_ptr, num_bytes, 0, 0, omp_get_initial_device(),
                    this->_dev_info.device_id);
}

void OmpOffloadExecutor::spmv_init(CSRSpMV<float>& op, CSRMatrix<float>& mat,
                                   bool symmetric)
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
          mat.values(), symmetric, *this);
}

void OmpOffloadExecutor::spmv_init(CSRSpMV<double>& op, CSRMatrix<double>& mat,
                                   bool symmetric)
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
          mat.values(), symmetric, *this);
}

void OmpOffloadExecutor::spmv_run(const CSRSpMV<float>& op,
                                  const CSRMatrix<float>& mat, float alpha,
                                  float* __restrict__ in, float beta,
                                  float* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
         mat.values(), mat.diagonal(), alpha, in, beta, out, *this);
}

void OmpOffloadExecutor::spmv_run(const CSRSpMV<double>& op,
                                  const CSRMatrix<double>& mat, double alpha,
                                  double* __restrict__ in, double beta,
                                  double* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
         mat.values(), mat.diagonal(), alpha, in, beta, out, *this);
}

void OmpOffloadExecutor::spmv_finalize(const CSRSpMV<float>& op) const
{
  op.finalize(*this);
}

void OmpOffloadExecutor::spmv_finalize(const CSRSpMV<double>& op) const
{
  op.finalize(*this);
}

void OmpOffloadExecutor::spmv_init(COOSpMV<float>& op, COOMatrix<float>& mat) {}

void OmpOffloadExecutor::spmv_init(COOSpMV<double>& op, COOMatrix<double>& mat)
{
}

void OmpOffloadExecutor::spmv_run(const COOSpMV<float>& op,
                                  const COOMatrix<float>& mat, float alpha,
                                  float* __restrict__ in, float beta,
                                  float* __restrict__ out) const
{
}

void OmpOffloadExecutor::spmv_run(const COOSpMV<double>& op,
                                  const COOMatrix<double>& mat, double alpha,
                                  double* __restrict__ in, double beta,
                                  double* __restrict__ out) const
{
}

void OmpOffloadExecutor::spmv_finalize(const COOSpMV<float>& op) const {}

void OmpOffloadExecutor::spmv_finalize(const COOSpMV<double>& op) const {}

void OmpOffloadExecutor::gather_ghosts_run(int num_indices,
                                           const int32_t* indices,
                                           const float* in, float* out) const
{
  #pragma omp target teams distribute parallel for	\
    is_device_ptr(indices)				\
    is_device_ptr(in)					\
    is_device_ptr(out)
  for (int i = 0; i < num_indices; ++i)
    out[i] = in[indices[i]];
}

void OmpOffloadExecutor::gather_ghosts_run(int num_indices,
                                           const int32_t* indices,
                                           const double* in, double* out) const
{
  #pragma omp target teams distribute parallel for	\
    is_device_ptr(indices)				\
    is_device_ptr(in)					\
    is_device_ptr(out)
  for (int i = 0; i < num_indices; ++i)
    out[i] = in[indices[i]];
}

} // namespace spmv

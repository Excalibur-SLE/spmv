// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "openacc_executor.h"
#include <cstring>
#include <typeinfo>

#include "coo_matrix.h"
#include "csr_matrix.h"

namespace spmv
{

OpenaccExecutor::OpenaccExecutor(int device_id)
{
  acc_init(acc_device_nvidia);
  acc_set_device_num(device_id, acc_device_nvidia);
}

OpenaccExecutor::~OpenaccExecutor()
{
  acc_shutdown(acc_device_nvidia);
}

void OpenaccExecutor::synchronize() const { acc_wait_all(); }

const DeviceExecutor& OpenaccExecutor::get_host() const { return *this; }

int OpenaccExecutor::get_num_devices() const
{
  return acc_get_num_devices(acc_device_nvidia);
}
  
int OpenaccExecutor::get_num_cus() const
{
  return acc_get_num_devices(acc_device_nvidia);
}

void* OpenaccExecutor::_alloc(size_t num_bytes) const
{
  void* ptr = nullptr;
  ptr = acc_malloc(num_bytes);
  return ptr;
}

void OpenaccExecutor::_free(void* ptr) const { acc_free(ptr); }

void OpenaccExecutor::_memset(void* ptr, int value, size_t num_bytes) const
{
};

void OpenaccExecutor::_copy(void* dst_ptr, const void* src_ptr,
                               size_t num_bytes) const
{
  // FIXME: not implemented in NVHPC
  //  acc_memcpy_device(dst_ptr, src_ptr, num_bytes);
}

void OpenaccExecutor::_copy_async(void* dst_ptr, const void* src_ptr,
				  size_t num_bytes, void* obj) const
{
  // FIXME: not implemented in NVHPC
  //  acc_memcpy_device(dst_ptr, src_ptr, num_bytes);
}
  
void OpenaccExecutor::_copy_from(void* dst_ptr,
                                    const DeviceExecutor& src_exec,
                                    const void* src_ptr, size_t num_bytes) const
{
  acc_memcpy_to_device(dst_ptr, const_cast<void*>(src_ptr), num_bytes);
}

void OpenaccExecutor::_copy_to(void* dst_ptr, const DeviceExecutor& dst_exec,
                                  const void* src_ptr, size_t num_bytes) const
{
  acc_memcpy_from_device(dst_ptr, const_cast<void*>(src_ptr), num_bytes);
}
  
void OpenaccExecutor::spmv_init(CSRSpMV<float>& op, CSRMatrix<float>& mat,
                                   bool symmetric)
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
          mat.values(), symmetric, *this);
}

void OpenaccExecutor::spmv_init(CSRSpMV<double>& op, CSRMatrix<double>& mat,
                                   bool symmetric)
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
          mat.values(), symmetric, *this);
}

void OpenaccExecutor::spmv_run(const CSRSpMV<float>& op,
                                  const CSRMatrix<float>& mat, float alpha,
                                  float* __restrict__ in, float beta,
                                  float* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
         mat.values(), mat.diagonal(), alpha, in, beta, out, *this);
}

void OpenaccExecutor::spmv_run(const CSRSpMV<double>& op,
                                  const CSRMatrix<double>& mat, double alpha,
                                  double* __restrict__ in, double beta,
                                  double* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
         mat.values(), mat.diagonal(), alpha, in, beta, out, *this);
}

void OpenaccExecutor::spmv_finalize(const CSRSpMV<float>& op) const
{
  op.finalize(*this);
}

void OpenaccExecutor::spmv_finalize(const CSRSpMV<double>& op) const
{
  op.finalize(*this);
}

void OpenaccExecutor::spmv_init(COOSpMV<float>& op, COOMatrix<float>& mat) {}

void OpenaccExecutor::spmv_init(COOSpMV<double>& op, COOMatrix<double>& mat)
{
}

void OpenaccExecutor::spmv_run(const COOSpMV<float>& op,
                                  const COOMatrix<float>& mat, float alpha,
                                  float* __restrict__ in, float beta,
                                  float* __restrict__ out) const
{
}

void OpenaccExecutor::spmv_run(const COOSpMV<double>& op,
                                  const COOMatrix<double>& mat, double alpha,
                                  double* __restrict__ in, double beta,
                                  double* __restrict__ out) const
{
}

void OpenaccExecutor::spmv_finalize(const COOSpMV<float>& op) const {}

void OpenaccExecutor::spmv_finalize(const COOSpMV<double>& op) const {}

void OpenaccExecutor::gather_ghosts_run(int num_indices,
                                           const int32_t* indices,
                                           const float* in, float* out) const
{
  #pragma acc kernels
  for (int i = 0; i < num_indices; ++i)
    out[i] = in[indices[i]];
}

void OpenaccExecutor::gather_ghosts_run(int num_indices,
                                           const int32_t* indices,
                                           const double* in, double* out) const
{
  #pragma acc kernels
  for (int i = 0; i < num_indices; ++i)
    out[i] = in[indices[i]];
}

} // namespace spmv

// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "omp_executor.h"

#include "coo_matrix.h"
#include "csr_matrix.h"

#include <cstring>
#include <iostream>
#include <typeinfo>

namespace spmv
{

void OmpExecutor::synchronize() const {}

const DeviceExecutor& OmpExecutor::get_host() const { return *this; }

int OmpExecutor::get_num_devices() const { return 1; }

int OmpExecutor::get_num_cus() const
{
  const char* threads_env = getenv("OMP_NUM_THREADS");
  int ret = 1;

  if (threads_env) {
    ret = atoi(threads_env);
    if (ret < 0)
      ret = 1;
  }

  return ret;
}

void* OmpExecutor::_alloc(size_t num_bytes) const
{
  // FIXME use OpenMP-5 allocators
  void* ptr = nullptr;
  ptr = std::malloc(num_bytes);
  // ptr = omp_alloc(num_bytes, omp_default_mem_alloc);
  return ptr;
}

void OmpExecutor::_free(void* ptr) const
{
  std::free(ptr);
  // omp_free(ptr, omp_default_mem_alloc);
}

void OmpExecutor::_memset(void* ptr, int value, size_t num_bytes) const
{
  // FIXME: parallelize
  std::memset(ptr, value, num_bytes);
}

void OmpExecutor::_copy(void* dst_ptr, const void* src_ptr,
                        size_t num_bytes) const
{
  if (num_bytes > 0) {
    // FIXME: parallelize
    std::memcpy(dst_ptr, src_ptr, num_bytes);
  }
}

void OmpExecutor::_copy_async(void* dst_ptr, const void* src_ptr,
                              size_t num_bytes, void* obj) const
{
  _copy(dst_ptr, src_ptr, num_bytes);
}

void OmpExecutor::_copy_from(void* dst_ptr, const DeviceExecutor& src_exec,
                             const void* src_ptr, size_t num_bytes) const
{
  if (typeid(src_exec) == typeid(OmpExecutor)) {
    if (num_bytes > 0) {
      // FIXME: parallelize
      std::memcpy(dst_ptr, src_ptr, num_bytes);
    }
  }
}

void OmpExecutor::_copy_to(void* dst_ptr, const DeviceExecutor& dst_exec,
                           const void* src_ptr, size_t num_bytes) const
{
}

void OmpExecutor::spmv_init(CSRSpMV<float>& op,
                            const CSRMatrix<float>& mat) const
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
          mat.values(), mat.symmetric(), *this);
}

void OmpExecutor::spmv_init(CSRSpMV<double>& op,
                            const CSRMatrix<double>& mat) const
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
          mat.values(), mat.symmetric(), *this);
}

void OmpExecutor::spmv_run(const CSRSpMV<float>& op,
                           const CSRMatrix<float>& mat, float alpha,
                           float* __restrict__ in, float beta,
                           float* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
         mat.values(), mat.diagonal(), alpha, in, beta, out, *this);
}

void OmpExecutor::spmv_run(const CSRSpMV<double>& op,
                           const CSRMatrix<double>& mat, double alpha,
                           double* __restrict__ in, double beta,
                           double* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
         mat.values(), mat.diagonal(), alpha, in, beta, out, *this);
}

void OmpExecutor::spmv_finalize(CSRSpMV<float>& op) const
{
  op.finalize(*this);
}

void OmpExecutor::spmv_finalize(CSRSpMV<double>& op) const
{
  op.finalize(*this);
}

void OmpExecutor::gather_ghosts_run(int num_indices, const int32_t* indices,
                                    const float* in, float* out) const
{
#pragma omp parallel for
  for (int i = 0; i < num_indices; ++i)
    out[i] = in[indices[i]];
}

void OmpExecutor::gather_ghosts_run(int num_indices, const int32_t* indices,
                                    const double* in, double* out) const
{
#pragma omp parallel for
  for (int i = 0; i < num_indices; ++i)
    out[i] = in[indices[i]];
}

} // namespace spmv

// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "reference_executor.h"

#include "coo_matrix.h"
#include "csr_matrix.h"

#include <cstring>
#include <iostream>
#include <typeinfo>

namespace spmv
{

void ReferenceExecutor::synchronize() const {}

const DeviceExecutor& ReferenceExecutor::get_host() const { return *this; }

void* ReferenceExecutor::_alloc(size_t num_bytes) const
{
  void* ptr = nullptr;
  ptr = std::malloc(num_bytes);
  return ptr;
}

void ReferenceExecutor::_free(void* ptr) const { std::free(ptr); }

void ReferenceExecutor::_memset(void* ptr, int value, size_t num_bytes) const
{
  std::memset(ptr, value, num_bytes);
}

void ReferenceExecutor::_copy(void* dst_ptr, const void* src_ptr,
                              size_t num_bytes) const
{
  if (num_bytes > 0) {
    if (dst_ptr == src_ptr)
      return;
    std::memcpy(dst_ptr, src_ptr, num_bytes);
  }
}

void ReferenceExecutor::_copy_async(void* dst_ptr, const void* src_ptr,
                                    size_t num_bytes, void* obj) const
{
  _copy(dst_ptr, src_ptr, num_bytes);
}

void ReferenceExecutor::_copy_from(void* dst_ptr,
                                   const DeviceExecutor& src_exec,
                                   const void* src_ptr, size_t num_bytes) const
{
  if (typeid(src_exec) == typeid(ReferenceExecutor)) {
    if (num_bytes > 0) {
      std::memcpy(dst_ptr, src_ptr, num_bytes);
    }
  }
}

void ReferenceExecutor::spmv_init(CSRSpMV<float>& op,
                                  const CSRMatrix<float>& mat) const
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
          mat.values(), mat.symmetric(), *this);
}

void ReferenceExecutor::spmv_init(CSRSpMV<double>& op,
                                  const CSRMatrix<double>& mat) const
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
          mat.values(), mat.symmetric(), *this);
}

void ReferenceExecutor::spmv_run(const CSRSpMV<float>& op,
                                 const CSRMatrix<float>& mat, float alpha,
                                 float* __restrict__ in, float beta,
                                 float* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
         mat.values(), mat.diagonal(), alpha, in, beta, out, *this);
}

void ReferenceExecutor::spmv_run(const CSRSpMV<double>& op,
                                 const CSRMatrix<double>& mat, double alpha,
                                 double* __restrict__ in, double beta,
                                 double* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
         mat.values(), mat.diagonal(), alpha, in, beta, out, *this);
}

void ReferenceExecutor::spmv_finalize(CSRSpMV<float>& op) const
{
  op.finalize(*this);
}

void ReferenceExecutor::spmv_finalize(CSRSpMV<double>& op) const
{
  op.finalize(*this);
}

void ReferenceExecutor::_copy_to(void* dst_ptr, const DeviceExecutor& dst_exec,
                                 const void* src_ptr, size_t num_bytes) const
{
}

void ReferenceExecutor::spmv_init(COOSpMV<float>& op,
                                  const COOMatrix<float>& mat) const
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowind(), mat.colind(),
          mat.values(), *this);
}

void ReferenceExecutor::spmv_init(COOSpMV<double>& op,
                                  const COOMatrix<double>& mat) const
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowind(), mat.colind(),
          mat.values(), *this);
}

void ReferenceExecutor::spmv_run(const COOSpMV<float>& op,
                                 const COOMatrix<float>& mat, float alpha,
                                 float* __restrict__ in, float beta,
                                 float* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowind(), mat.colind(),
         mat.values(), alpha, in, beta, out, *this);
}

void ReferenceExecutor::spmv_run(const COOSpMV<double>& op,
                                 const COOMatrix<double>& mat, double alpha,
                                 double* __restrict__ in, double beta,
                                 double* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowind(), mat.colind(),
         mat.values(), alpha, in, beta, out, *this);
}

void ReferenceExecutor::spmv_finalize(COOSpMV<float>& op) const
{
  op.finalize(*this);
}

void ReferenceExecutor::spmv_finalize(COOSpMV<double>& op) const
{
  op.finalize(*this);
}

void ReferenceExecutor::gather_ghosts_run(int num_indices,
                                          const int32_t* indices,
                                          const float* in, float* out) const
{
  for (int i = 0; i < num_indices; ++i)
    out[i] = in[indices[i]];
}

void ReferenceExecutor::gather_ghosts_run(int num_indices,
                                          const int32_t* indices,
                                          const double* in, double* out) const
{
  for (int i = 0; i < num_indices; ++i)
    out[i] = in[indices[i]];
}

} // namespace spmv

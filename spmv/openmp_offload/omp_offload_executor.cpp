// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "omp_offload_executor.h"
#include <cstring>
#include <typeinfo>

#include "coo_matrix.h"
#include "csr_matrix.h"

namespace spmv
{

void OmpOffloadExecutor::synchronize() const {}

const DeviceExecutor& OmpOffloadExecutor::get_host() const { return *this; }

int OmpOffloadExecutor::get_num_cus() const
{
  // FIXME
  return 1;
  // const char* threads_env = getenv("OMP_NUM_THREADS");
  // int ret = 1;

  // if (threads_env) {
  //   ret = atoi(threads_env);
  //   if (ret < 0)
  //     ret = 1;
  // }

  // return ret;
}

void* OmpOffloadExecutor::_alloc(size_t num_bytes) const
{
  // FIXME use OpenMP-5 allocators
  void* ptr = nullptr;
  ptr = std::malloc(num_bytes);
  return ptr;
}

void OmpOffloadExecutor::_free(void* ptr) const { std::free(ptr); }

void OmpOffloadExecutor::_memset(void* ptr, int value, size_t num_bytes) const
{
  std::memset(ptr, value, num_bytes);
};

void OmpOffloadExecutor::_copy(void* dst_ptr, const void* src_ptr,
                               size_t num_bytes) const
{
  if (num_bytes > 0) {
    std::memcpy(dst_ptr, src_ptr, num_bytes);
  }
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
  if (typeid(src_exec) == typeid(OmpOffloadExecutor)) {
    if (num_bytes > 0) {
      //      #pragma omp target enter data map(to : rowptr[:nrows + 1])
      std::memcpy(dst_ptr, src_ptr, num_bytes);
    }
  }
}

void OmpOffloadExecutor::_copy_to(void* dst_ptr, const DeviceExecutor& dst_exec,
                                  const void* src_ptr, size_t num_bytes) const
{
  //  #pragma omp target exit data map(release : rowptr[:nrows + 1])
  if (typeid(dst_exec) == typeid(OmpOffloadExecutor)) {
    if (num_bytes > 0) {
      //      #pragma omp target enter data map(to : rowptr[:nrows + 1])
      std::memcpy(dst_ptr, src_ptr, num_bytes);
    }
  }
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
  for (int i = 0; i < num_indices; ++i)
    out[i] = in[indices[i]];
}

void OmpOffloadExecutor::gather_ghosts_run(int num_indices,
                                           const int32_t* indices,
                                           const double* in, double* out) const
{
  for (int i = 0; i < num_indices; ++i)
    out[i] = in[indices[i]];
}

} // namespace spmv

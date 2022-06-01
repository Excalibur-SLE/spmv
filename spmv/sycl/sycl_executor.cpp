// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "sycl_executor.h"

#include "coo_matrix.h"
#include "csr_matrix.h"

#include <cstring>
#include <typeinfo>

namespace spmv
{

SyclExecutor::SyclExecutor(sycl::queue* q) : _queue(q)
{
  auto device = q->get_device();
  if (device.is_cpu()) {
    this->_dev_info.type = DeviceType::cpu;
  } else if (device.is_gpu()) {
    this->_dev_info.type = DeviceType::gpu;
  } else {
    this->_dev_info.type = DeviceType::undefined;
  }
}

void SyclExecutor::synchronize() const { _queue->wait(); }

const DeviceExecutor& SyclExecutor::get_host() const { return *this; }

int SyclExecutor::get_num_devices() const
{
  auto platforms = sycl::platform::get_platforms();
  int num_devices = 0;
  for (auto& e : platforms) {
    num_devices += e.get_devices(sycl::info::device_type::all).size();
  }
  return num_devices;
}

int SyclExecutor::get_num_cus() const
{
#ifdef __HIPSYCL__
  const char* threads_env = getenv("OMP_NUM_THREADS");
#else
  const char* threads_env = getenv("DPCPP_CPU_NUM_CUS");
#endif
  int ret = 1;

  if (threads_env) {
    ret = atoi(threads_env);
    if (ret < 0)
      ret = 1;
  }

  return ret;
}

void* SyclExecutor::_alloc(size_t num_bytes) const
{
  void* ptr = nullptr;
  ptr = sycl::malloc_shared(num_bytes, *_queue);
  return ptr;
}

void SyclExecutor::_free(void* ptr) const
{
  if (ptr != nullptr)
    sycl::free(ptr, *_queue);
}

void SyclExecutor::_memset(void* ptr, int value, size_t num_bytes) const
{
  _queue->memset(ptr, value, num_bytes);
};

void SyclExecutor::_copy(void* dst_ptr, const void* src_ptr,
                         size_t num_bytes) const
{
  _queue->memcpy((char*)dst_ptr, (char*)src_ptr, num_bytes).wait();
}

void SyclExecutor::_copy_async(void* dst_ptr, const void* src_ptr,
                               size_t num_bytes, void* obj) const
{
  _queue->copy((char*)src_ptr, (char*)dst_ptr, num_bytes);
}

void SyclExecutor::_copy_from(void* dst_ptr, const DeviceExecutor& src_exec,
                              const void* src_ptr, size_t num_bytes) const
{
  if (typeid(src_exec) == typeid(SyclExecutor)) {
    _queue->copy((char*)src_ptr, (char*)dst_ptr, num_bytes).wait();
  }
}

void SyclExecutor::_copy_to(void* dst_ptr, const DeviceExecutor& dst_exec,
                            const void* src_ptr, size_t num_bytes) const
{
  if (typeid(dst_exec) == typeid(SyclExecutor)) {
    _queue->copy((char*)src_ptr, (char*)dst_ptr, num_bytes).wait();
  }
}

void SyclExecutor::spmv_init(CSRSpMV<float>& op,
                             const CSRMatrix<float>& mat) const
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
          mat.values(), mat.symmetric(), *this);
}

void SyclExecutor::spmv_init(CSRSpMV<double>& op,
                             const CSRMatrix<double>& mat) const
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
          mat.values(), mat.symmetric(), *this);
}

void SyclExecutor::spmv_run(const CSRSpMV<float>& op,
                            const CSRMatrix<float>& mat, float alpha,
                            float* __restrict__ in, float beta,
                            float* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
         mat.values(), mat.diagonal(), alpha, in, beta, out, *this);
}

void SyclExecutor::spmv_run(const CSRSpMV<double>& op,
                            const CSRMatrix<double>& mat, double alpha,
                            double* __restrict__ in, double beta,
                            double* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
         mat.values(), mat.diagonal(), alpha, in, beta, out, *this);
}

void SyclExecutor::spmv_finalize(CSRSpMV<float>& op) const
{
  op.finalize(*this);
}

void SyclExecutor::spmv_finalize(CSRSpMV<double>& op) const
{
  op.finalize(*this);
}

void SyclExecutor::gather_ghosts_run(int num_indices, const int32_t* indices,
                                     const float* in, float* out) const
{
  _queue
      ->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range(num_indices), [=](sycl::id<1> it) {
          const int tid = it[0];
          out[tid] = in[indices[tid]];
        });
      })
      .wait();
}

void SyclExecutor::gather_ghosts_run(int num_indices, const int32_t* indices,
                                     const double* in, double* out) const
{
  _queue
      ->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range(num_indices), [=](sycl::id<1> it) {
          const int tid = it[0];
          out[tid] = in[indices[tid]];
        });
      })
      .wait();
}

} // namespace spmv

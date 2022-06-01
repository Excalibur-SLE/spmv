// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include "config.h"
#include "spmv_export.h"

#include "cuda_helper.h"
#include "device_executor.h"

#include <memory>

namespace spmv
{

class SPMV_EXPORT CudaExecutor final : public DeviceExecutor
{
public:
  ~CudaExecutor();

  // Factory function
  template <typename... Ts>
  static std::unique_ptr<CudaExecutor> create(Ts&&... params)
  {
    // Can't use make_unique with private ctors
    std::unique_ptr<CudaExecutor> ptr(nullptr);
    ptr.reset(new CudaExecutor(std::forward<Ts>(params)...));
    return ptr;
  }

  void synchronize() const override;
  const DeviceExecutor& get_host() const { return *_host; }
  int get_num_devices() const override;
  int get_num_cus() const override;

  // CSR format
  void spmv_init(CSRSpMV<float>& op,
                 const CSRMatrix<float>& mat) const override;
  void spmv_init(CSRSpMV<double>& op,
                 const CSRMatrix<double>& mat) const override;
  void spmv_run(const CSRSpMV<float>& op, const CSRMatrix<float>& mat,
                float alpha, float* __restrict__ in, float beta,
                float* __restrict__ out) const override;
  void spmv_run(const CSRSpMV<double>& op, const CSRMatrix<double>& mat,
                double alpha, double* __restrict__ in, double beta,
                double* __restrict__ out) const override;
  void spmv_finalize(CSRSpMV<float>& op) const override;
  void spmv_finalize(CSRSpMV<double>& op) const override;

  // COO format
  void spmv_init(COOSpMV<float>& op,
                 const COOMatrix<float>& mat) const override;
  void spmv_init(COOSpMV<double>& op,
                 const COOMatrix<double>& mat) const override;
  void spmv_run(const COOSpMV<float>& op, const COOMatrix<float>& mat,
                float alpha, float* __restrict__ in, float beta,
                float* __restrict__ out) const override;
  void spmv_run(const COOSpMV<double>& op, const COOMatrix<double>& mat,
                double alpha, double* __restrict__ in, double beta,
                double* __restrict__ out) const override;
  void spmv_finalize(COOSpMV<float>& op) const override;
  void spmv_finalize(COOSpMV<double>& op) const override;

  // Gather ghosts
  void gather_ghosts_run(int num_indices, const int32_t* indices,
                         const float* in, float* out) const override;
  void gather_ghosts_run(int num_indices, const int32_t* indices,
                         const double* in, double* out) const override;

  // Extended API
  void set_cuda_stream(cudaStream_t stream);
  void reset_cuda_stream();
  cudaStream_t get_cuda_stream() const { return _stream; }
  cublasHandle_t get_cublas_handle() const { return _cublas_handle; }
  cusparseHandle_t get_cusparse_handle() const { return _cusparse_handle; }

protected:
  // Inherited API
  // FIXME support multiple allocation modes
  void* _alloc(size_t num_bytes) const override;
  void _free(void* ptr) const override;
  void _memset(void* ptr, int value, size_t num_bytes) const override;
  void _copy(void* dst_ptr, const void* src_ptr,
             size_t num_bytes) const override;
  void _copy_async(void* dst_ptr, const void* src_ptr, size_t num_bytes,
                   void* obj) const override;
  void _copy_from(void* dst_ptr, const DeviceExecutor& src_exec,
                  const void* src_ptr, size_t num_bytes) const;
  void _copy_to(void* dst_ptr, const DeviceExecutor& dst_exec,
                const void* src_ptr, size_t num_bytes) const;

private:
  CudaExecutor(int device_id, std::shared_ptr<DeviceExecutor> host);

private:
  std::shared_ptr<DeviceExecutor> _host = nullptr;
  cublasHandle_t _cublas_handle = nullptr;
  cusparseHandle_t _cusparse_handle = nullptr;
  cudaStream_t _stream = nullptr;
};

} // namespace spmv

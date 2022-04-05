// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include "coo_kernels.h"
#include "csr_kernels.h"
#include "device_executor.h"

namespace spmv
{

class SPMV_EXPORT ReferenceExecutor : public DeviceExecutor
{
public:
  ~ReferenceExecutor(){};

  // Factory function
  template <typename... Ts>
  static std::unique_ptr<ReferenceExecutor> create(Ts&&... params)
  {
    // Can't use make_unique with private ctors
    std::unique_ptr<ReferenceExecutor> ptr(nullptr);
    ptr.reset(new ReferenceExecutor(std::forward<Ts>(params)...));
    return ptr;
  }

  void synchronize() const override;
  const DeviceExecutor& get_host() const override;
  int get_num_cus() const override { return 1; };
  int get_num_devices() const override { return 1; };

  // COO format
  void spmv_init(COOSpMV<float>& op, COOMatrix<float>& mat) override;
  void spmv_init(COOSpMV<double>& op, COOMatrix<double>& mat) override;
  void spmv_run(const COOSpMV<float>& op, const COOMatrix<float>& mat,
                float alpha, float* __restrict__ in, float beta,
                float* __restrict__ out) const override;
  void spmv_run(const COOSpMV<double>& op, const COOMatrix<double>& mat,
                double alpha, double* __restrict__ in, double beta,
                double* __restrict__ out) const override;
  void spmv_finalize(const COOSpMV<float>& op) const override;
  void spmv_finalize(const COOSpMV<double>& op) const override;

  // CSR format
  void spmv_init(CSRSpMV<float>& op, CSRMatrix<float>& mat,
                 bool symmetric) override;
  void spmv_init(CSRSpMV<double>& op, CSRMatrix<double>& mat,
                 bool symmetric) override;
  void spmv_run(const CSRSpMV<float>& op, const CSRMatrix<float>& mat,
                float alpha, float* __restrict__ in, float beta,
                float* __restrict__ out) const override;
  void spmv_run(const CSRSpMV<double>& op, const CSRMatrix<double>& mat,
                double alpha, double* __restrict__ in, double beta,
                double* __restrict__ out) const override;
  void spmv_finalize(const CSRSpMV<float>& op) const override;
  void spmv_finalize(const CSRSpMV<double>& op) const override;

  // Gather ghosts
  void gather_ghosts_run(int num_indices, const int32_t* indices,
                         const float* in, float* out) const override;
  void gather_ghosts_run(int num_indices, const int32_t* indices,
                         const double* in, double* out) const override;

protected:
  void* _alloc(size_t num_bytes) const override;
  void _free(void* ptr) const override;
  void _memset(void* ptr, int value, size_t num_bytes) const override;
  void _copy(void* dst_ptr, const void* src_ptr,
             size_t num_bytes) const override;
  void _copy_async(void* dst_ptr, const void* src_ptr, size_t num_bytes,
                   void* obj) const override;
  void _copy_from(void* dst_ptr, const DeviceExecutor& src_exec,
                  const void* src_ptr, size_t num_bytes) const override;
  void _copy_to(void* dst_ptr, const DeviceExecutor& dst_exec,
                const void* src_ptr, size_t num_bytes) const override;

private:
  ReferenceExecutor() = default;
};

} // namespace spmv

// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include "config.h"
#include "spmv_export.h"

#include <cstdlib>
#include <memory>

namespace spmv
{

// Forward declare all SpMV classes
template <typename T>
class COOMatrix;
template <typename T>
class COOSpMV;

template <typename T>
class CSRMatrix;
template <typename T>
class CSRSpMV;

enum class DeviceType { undefined, cpu, gpu };

// Abstract polymorphic base class for handling a device
class SPMV_EXPORT DeviceExecutor
{
public:
  /* DeviceExecutor(DeviceExecutor&) = delete; */
  /* DeviceExecutor(DeviceExecutor&&) = default; */
  /* DeviceExecutor& operator=(DeviceExecutor&) = delete; */
  /* DeviceExecutor& operator=(DeviceExecutor&&) = default; */
  virtual ~DeviceExecutor() = default;

  template <typename T>
  T* alloc(size_t num_elems) const
  {
    T* ptr = nullptr;
    ptr = static_cast<T*>(_alloc(num_elems * sizeof(T)));
    return ptr;
  }

  template <typename T>
  void memset(T* ptr, int value, size_t num_elems) const
  {
    _memset(ptr, value, num_elems * sizeof(T));
  }

  void free(void* ptr) const;

  template <typename T>
  void copy(T* dst_ptr, const T* src_ptr, size_t num_elems) const
  {
    _copy(dst_ptr, src_ptr, num_elems * sizeof(T));
  }

  // FIXME use opaque pointer for stream
  template <typename T>
  void copy_async(T* dst_ptr, const T* src_ptr, size_t num_elems,
                  void* stream) const
  {
    _copy_async(dst_ptr, src_ptr, num_elems * sizeof(T), stream);
  }

  template <typename T>
  void copy_from(T* dst_ptr, const DeviceExecutor& src_exec, const T* src_ptr,
                 size_t num_elems) const
  {
    _copy_from(dst_ptr, src_exec, src_ptr, num_elems * sizeof(T));
  }

  template <typename T>
  void copy_to(T* dst_ptr, const DeviceExecutor& dst_exec, const T* src_ptr,
               size_t num_elems) const
  {
    _copy_to(dst_ptr, dst_exec, src_ptr, num_elems * sizeof(T));
  }

  virtual void synchronize() const = 0;
  virtual const DeviceExecutor& get_host() const = 0;
  virtual int get_num_devices() const = 0;
  virtual int get_num_cus() const = 0;

  // CSR format visitors
  virtual void spmv_init(CSRSpMV<float>& op,
                         const CSRMatrix<float>& mat) const = 0;
  virtual void spmv_init(CSRSpMV<double>& op,
                         const CSRMatrix<double>& mat) const = 0;
  virtual void spmv_run(const CSRSpMV<float>& op, const CSRMatrix<float>& mat,
                        float alpha, float* __restrict__ in, float beta,
                        float* __restrict__ out) const = 0;
  virtual void spmv_run(const CSRSpMV<double>& op, const CSRMatrix<double>& mat,
                        double alpha, double* __restrict__ in, double beta,
                        double* __restrict__ out) const = 0;
  virtual void spmv_finalize(CSRSpMV<float>& op) const = 0;
  virtual void spmv_finalize(CSRSpMV<double>& op) const = 0;

  // COO format visitors
  virtual void spmv_init(COOSpMV<float>& op, const COOMatrix<float>& mat) const
  {
  }
  virtual void spmv_init(COOSpMV<double>& op,
                         const COOMatrix<double>& mat) const
  {
  }
  virtual void spmv_run(const COOSpMV<float>& op, const COOMatrix<float>& mat,
                        float alpha, float* __restrict__ in, float beta,
                        float* __restrict__ out) const
  {
  }
  virtual void spmv_run(const COOSpMV<double>& op, const COOMatrix<double>& mat,
                        double alpha, double* __restrict__ in, double beta,
                        double* __restrict__ out) const
  {
  }
  virtual void spmv_finalize(COOSpMV<float>& op) const {}
  virtual void spmv_finalize(COOSpMV<double>& op) const {}

  // Gather ghosts
  virtual void gather_ghosts_run(int num_indices, const int32_t* indices,
                                 const float* in, float* out) const = 0;
  virtual void gather_ghosts_run(int num_indices, const int32_t* indices,
                                 const double* in, double* out) const = 0;

protected:
  virtual void* _alloc(size_t num_bytes) const = 0;
  virtual void _free(void* ptr) const = 0;
  virtual void _memset(void* ptr, int value, size_t num_bytes) const = 0;
  virtual void _copy(void* dst_ptr, const void* src_ptr,
                     size_t num_bytes) const = 0;
  virtual void _copy_async(void* dst_ptr, const void* src_ptr, size_t num_bytes,
                           void* obj) const = 0;
  virtual void _copy_from(void* dst_ptr, const DeviceExecutor& src_exec,
                          const void* src_ptr, size_t num_bytes) const = 0;
  virtual void _copy_to(void* dst_ptr, const DeviceExecutor& dst_exec,
                        const void* src_ptr, size_t num_bytes) const = 0;

  struct DeviceInfo {
    DeviceType type = DeviceType::undefined;
    int id = -1;
  };
  DeviceInfo _dev_info;

public:
  const DeviceInfo& get_device_info() const { return _dev_info; }
  DeviceType get_device_type() const { return _dev_info.type; }
  int get_device_id() const { return _dev_info.id; }
};

} // namespace spmv

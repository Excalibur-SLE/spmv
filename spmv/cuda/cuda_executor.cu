// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "coo_matrix.h"
#include "csr_matrix.h"
#include "cuda_executor.h"
#include "reference_executor.h"

namespace spmv
{

CudaExecutor::CudaExecutor(int device_id, std::shared_ptr<DeviceExecutor> host)
    : _host(host)
{
  CHECK_CUDA(cudaSetDevice(device_id));
  this->_dev_info.device_id = device_id;
  CHECK_CUBLAS(cublasCreate(&_cublas_handle));
  CHECK_CUSPARSE(cusparseCreate(&_cusparse_handle));
}

CudaExecutor::~CudaExecutor()
{
  CHECK_CUSPARSE(cusparseDestroy(_cusparse_handle));
  CHECK_CUBLAS(cublasDestroy(_cublas_handle));
}

void CudaExecutor::synchronize() const
{
  if (_stream) {
    CHECK_CUDA(cudaStreamSynchronize(_stream));
  } else {
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}

int CudaExecutor::get_num_devices() const
{
  int num_devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&num_devices));
  return num_devices;
}

int CudaExecutor::get_num_cus() const
{
  cudaDeviceProp device_prop;
  CHECK_CUDA(
      cudaGetDeviceProperties(&device_prop, this->get_device_info().device_id));
  return device_prop.multiProcessorCount;
}

void* CudaExecutor::_alloc(size_t num_bytes) const
{
  void* ptr = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&ptr, num_bytes));
  return ptr;
}

void CudaExecutor::_free(void* ptr) const { CHECK_CUDA(cudaFree(ptr)); }

void CudaExecutor::_memset(void* ptr, int value, size_t num_bytes) const
{
  CHECK_CUDA(cudaMemset(ptr, value, num_bytes));
};

void CudaExecutor::_copy(void* dst_ptr, const void* src_ptr,
                         size_t num_bytes) const
{
  CHECK_CUDA(cudaMemcpy(dst_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToDevice));
}

void CudaExecutor::_copy_async(void* dst_ptr, const void* src_ptr,
                               size_t num_bytes, void* obj) const
{
  cudaStream_t* stream = reinterpret_cast<cudaStream_t*>(obj);
  CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, num_bytes,
                             cudaMemcpyDeviceToDevice, *stream));
}

void CudaExecutor::_copy_from(void* dst_ptr, const DeviceExecutor& src_exec,
                              const void* src_ptr, size_t num_bytes) const
{
  if (num_bytes > 0) {
    if (typeid(src_exec) == typeid(ReferenceExecutor)) {
      CHECK_CUDA(
          cudaMemcpy(dst_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice));
    } else if (typeid(src_exec) == typeid(CudaExecutor)) {
      CHECK_CUDA(cudaMemcpyPeer(dst_ptr, this->get_device_id(), src_ptr,
                                src_exec.get_device_id(), num_bytes));
    }
  }
}

void CudaExecutor::_copy_to(void* dst_ptr, const DeviceExecutor& dst_exec,
                            const void* src_ptr, size_t num_bytes) const
{
  if (num_bytes > 0) {
    if (typeid(dst_exec) == typeid(ReferenceExecutor)) {
      CHECK_CUDA(
          cudaMemcpy(dst_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToHost));
    }
  }
}

void CudaExecutor::spmv_init(CSRSpMV<float>& op, CSRMatrix<float>& mat,
                             bool symmetric)
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
          mat.values(), symmetric, *this);
}

void CudaExecutor::spmv_init(CSRSpMV<double>& op, CSRMatrix<double>& mat,
                             bool symmetric)
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
          mat.values(), symmetric, *this);
}

void CudaExecutor::spmv_run(const CSRSpMV<float>& op,
                            const CSRMatrix<float>& mat, float alpha,
                            float* __restrict__ in, float beta,
                            float* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
         mat.values(), mat.diagonal(), alpha, in, beta, out, *this);
}

void CudaExecutor::spmv_run(const CSRSpMV<double>& op,
                            const CSRMatrix<double>& mat, double alpha,
                            double* __restrict__ in, double beta,
                            double* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowptr(), mat.colind(),
         mat.values(), mat.diagonal(), alpha, in, beta, out, *this);
}

void CudaExecutor::spmv_finalize(const CSRSpMV<float>& op) const
{
  op.finalize(*this);
}

void CudaExecutor::spmv_finalize(const CSRSpMV<double>& op) const
{
  op.finalize(*this);
}

void CudaExecutor::spmv_init(COOSpMV<float>& op, COOMatrix<float>& mat)
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowind(), mat.colind(),
          mat.values(), *this);
}

void CudaExecutor::spmv_init(COOSpMV<double>& op, COOMatrix<double>& mat)
{
  op.init(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowind(), mat.colind(),
          mat.values(), *this);
}

void CudaExecutor::spmv_run(const COOSpMV<float>& op,
                            const COOMatrix<float>& mat, float alpha,
                            float* __restrict__ in, float beta,
                            float* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowind(), mat.colind(),
         mat.values(), alpha, in, beta, out, *this);
}

void CudaExecutor::spmv_run(const COOSpMV<double>& op,
                            const COOMatrix<double>& mat, double alpha,
                            double* __restrict__ in, double beta,
                            double* __restrict__ out) const
{
  op.run(mat.rows(), mat.cols(), mat.non_zeros(), mat.rowind(), mat.colind(),
         mat.values(), alpha, in, beta, out, *this);
}

void CudaExecutor::spmv_finalize(const COOSpMV<float>& op) const
{
  op.finalize(*this);
}

void CudaExecutor::spmv_finalize(const COOSpMV<double>& op) const
{
  op.finalize(*this);
}

void CudaExecutor::set_cuda_stream(cudaStream_t stream)
{
  _stream = stream;
  CHECK_CUBLAS(cublasSetStream(_cublas_handle, stream));
  CHECK_CUSPARSE(cusparseSetStream(_cusparse_handle, stream));
}

void CudaExecutor::reset_cuda_stream()
{
  _stream = nullptr;
  CHECK_CUBLAS(cublasSetStream(_cublas_handle, 0));
  CHECK_CUSPARSE(cusparseSetStream(_cusparse_handle, 0));
}

template <typename T>
__global__ void gather_ghosts(const int N, const int* indices, const T* in,
                              T* out)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    out[gid] = in[indices[gid]];
  }
}

void CudaExecutor::gather_ghosts_run(int num_indices, const int32_t* indices,
                                     const float* in, float* out) const
{
  const int block_size = 128;
  const int num_blocks = (num_indices + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  if (_stream == nullptr) {
    gather_ghosts<<<dimGrid, dimBlock>>>(num_indices, indices, in, out);
    cudaDeviceSynchronize();
  } else {
    gather_ghosts<<<dimGrid, dimBlock, 0, _stream>>>(num_indices, indices, in,
                                                     out);
    cudaStreamSynchronize(_stream);
  }
}

void CudaExecutor::gather_ghosts_run(int num_indices, const int32_t* indices,
                                     const double* in, double* out) const
{
  const int block_size = 128;
  const int num_blocks = (num_indices + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  if (_stream == nullptr) {
    gather_ghosts<<<dimGrid, dimBlock>>>(num_indices, indices, in, out);
    cudaDeviceSynchronize();
  } else {
    gather_ghosts<<<dimGrid, dimBlock, 0, _stream>>>(num_indices, indices, in,
                                                     out);
    cudaStreamSynchronize(_stream);
  }
}

} // namespace spmv

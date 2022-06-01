// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "cuda_executor.h"

namespace spmv
{

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
    CHECK_CUDA(cudaDeviceSynchronize());
  } else {
    gather_ghosts<<<dimGrid, dimBlock, 0, _stream>>>(num_indices, indices, in,
                                                     out);
    CHECK_CUDA(cudaStreamSynchronize(_stream));
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
    CHECK_CUDA(cudaDeviceSynchronize());
  } else {
    gather_ghosts<<<dimGrid, dimBlock, 0, _stream>>>(num_indices, indices, in,
                                                     out);
    CHECK_CUDA(cudaStreamSynchronize(_stream));
  }
}

} // namespace spmv

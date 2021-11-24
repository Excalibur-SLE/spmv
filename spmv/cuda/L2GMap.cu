// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "L2GMap.h"
#include "helper_cuda.h"
#include <cassert>
#include <iostream>

using namespace spmv;

//-----------------------------------------------------------------------------
template <typename T>
__global__ void gather_ghosts(const int N, const int* indices, const T* in,
                              T* out)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    out[gid] = in[indices[gid]];
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update(T* vec_data, cudaStream_t& stream) const
{
  switch (_cm) {
  case CommunicationModel::p2p_blocking:
    update_p2p(vec_data, stream);
    break;
  case CommunicationModel::p2p_nonblocking:
    update_p2p_start(vec_data, stream);
    break;
  case CommunicationModel::collective_blocking:
  case CommunicationModel::collective_nonblocking:
  case CommunicationModel::onesided_put_active:
  case CommunicationModel::onesided_put_passive:
    std::cout << "Unsupported communication model for GPUs." << std::endl;
    break;
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_finalise(T* vec_data, cudaStream_t& stream) const
{
  assert(_cm == CommunicationModel::p2p_nonblocking);
  update_p2p_end(vec_data, stream);
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_p2p(T* vec_data, cudaStream_t& stream) const
{
  assert(_cm == CommunicationModel::p2p_blocking);

  // Get data from local indices to send to other processes, landing in their
  // ghost region
  const int num_indices = _indexbuf.size();
  T* databuf = nullptr;
  // FIXME make persistent
  CHECK_CUDA(cudaMalloc((void**)&databuf, num_indices * sizeof(T)));
  const int block_size = 128;
  const int num_blocks = (num_indices + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  gather_ghosts<<<dimGrid, dimBlock, 0, stream>>>(num_indices, _d_indexbuf,
                                                  vec_data, databuf);
  cudaStreamSynchronize(stream);

  // Send actual values - NB meaning of _send and _recv count/offset is
  // reversed
  MPI_Datatype data_type = mpi_type<T>();
  const int num_neighbours = _neighbours.size();
  for (int i = 0; i < num_neighbours; ++i) {
    MPI_Irecv(vec_data + _send_offset[i], _send_count[i], data_type,
              _neighbours[i], 0, _comm, &(_req[i]));
  }

  for (int i = 0; i < num_neighbours; ++i) {
    MPI_Send(databuf + _recv_offset[i], _recv_count[i], data_type,
             _neighbours[i], 0, _comm);
  }

  MPI_Waitall(num_neighbours, _req, MPI_STATUSES_IGNORE);
  CHECK_CUDA(cudaFree(databuf));
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_p2p_start(T* vec_data, cudaStream_t& stream) const
{
  assert(_cm == CommunicationModel::p2p_nonblocking);
  const int num_indices = _indexbuf.size();

  // Allocate send and receive buffers, if this is the first call to update
  if (_send_buf_device == nullptr)
    CHECK_CUDA(cudaMalloc((void**)&_send_buf_device, num_indices * sizeof(T)));
  if (_recv_buf_device == nullptr)
    CHECK_CUDA(cudaMalloc((void**)&_recv_buf_device, num_ghosts() * sizeof(T)));

  // Get data from local indices to send to other processes, landing in their
  // ghost region
  const int block_size = 128;
  const int num_blocks = (num_indices + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  gather_ghosts<<<dimGrid, dimBlock, 0, stream>>>(
      num_indices, _d_indexbuf, vec_data, static_cast<T*>(_send_buf_device));
  cudaStreamSynchronize(stream);

  // Send actual values - NB meaning of _send and _recv count/offset is
  // reversed
  MPI_Datatype data_type = mpi_type<T>();
  const int num_neighbours = _neighbours.size();
  for (int i = 0; i < num_neighbours; ++i) {
    T* recv_buf = static_cast<T*>(_recv_buf_device) + _send_offset[i];
    MPI_Irecv(recv_buf, _send_count[i], data_type, _neighbours[i], 0, _comm,
              &(_req[i]));
  }

  for (int i = 0; i < num_neighbours; ++i) {
    T* send_buf = static_cast<T*>(_send_buf_device) + _recv_offset[i];
    MPI_Isend(send_buf, _recv_count[i], data_type, _neighbours[i], 0, _comm,
              &(_req[num_neighbours + i]));
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_p2p_end(T* vec_data, cudaStream_t& stream) const
{
  assert(_cm == CommunicationModel::p2p_nonblocking);
  MPI_Waitall(2 * _neighbours.size(), _req, MPI_STATUSES_IGNORE);
  // Copy ghosts from intermediate buffer to vector
  CHECK_CUDA(cudaMemcpyAsync(vec_data + local_size(), _recv_buf_device,
                             num_ghosts() * sizeof(T), cudaMemcpyDeviceToDevice,
                             stream));
}
//-----------------------------------------------------------------------------

// Explicit instantiation
template void spmv::L2GMap::update<float>(float*, cudaStream_t&) const;
template void spmv::L2GMap::update<double>(double*, cudaStream_t&) const;
template void spmv::L2GMap::update<std::complex<float>>(std::complex<float>*,
                                                        cudaStream_t&) const;
template void spmv::L2GMap::update<std::complex<double>>(std::complex<double>*,
                                                         cudaStream_t&) const;
template void spmv::L2GMap::update_finalise<float>(float*, cudaStream_t&) const;
template void spmv::L2GMap::update_finalise<double>(double*,
                                                    cudaStream_t&) const;
template void
spmv::L2GMap::update_finalise<std::complex<float>>(std::complex<float>*,
                                                   cudaStream_t&) const;
template void
spmv::L2GMap::update_finalise<std::complex<double>>(std::complex<double>*,
                                                    cudaStream_t&) const;
template void spmv::L2GMap::update_p2p<float>(float*, cudaStream_t&) const;
template void spmv::L2GMap::update_p2p<double>(double*, cudaStream_t&) const;
template void
spmv::L2GMap::update_p2p<std::complex<float>>(std::complex<float>*,
                                              cudaStream_t&) const;
template void
spmv::L2GMap::update_p2p<std::complex<double>>(std::complex<double>*,
                                               cudaStream_t&) const;
template void spmv::L2GMap::update_p2p_start<float>(float*,
                                                    cudaStream_t&) const;
template void spmv::L2GMap::update_p2p_start<double>(double*,
                                                     cudaStream_t&) const;
template void
spmv::L2GMap::update_p2p_start<std::complex<float>>(std::complex<float>*,
                                                    cudaStream_t&) const;
template void
spmv::L2GMap::update_p2p_start<std::complex<double>>(std::complex<double>*,
                                                     cudaStream_t&) const;
template void spmv::L2GMap::update_p2p_end<float>(float*, cudaStream_t&) const;
template void spmv::L2GMap::update_p2p_end<double>(double*,
                                                   cudaStream_t&) const;
template void
spmv::L2GMap::update_p2p_end<std::complex<float>>(std::complex<float>*,
                                                  cudaStream_t&) const;
template void
spmv::L2GMap::update_p2p_end<std::complex<double>>(std::complex<double>*,
                                                   cudaStream_t&) const;

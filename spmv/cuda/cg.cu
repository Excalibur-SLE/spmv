// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "L2GMap.h"
#include "Matrix.h"
#include "cg_cuda.h"
#include "helper_cuda.h"

#include <cublas_v2.h>

//-----------------------------------------------------------------------------
__global__ void compute_alpha(double* rnorm_old, double* pdotAp, double* alpha,
                              double* neg_alpha)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid == 0) {
    alpha[0] = (rnorm_old[0] * rnorm_old[0]) / pdotAp[0];
    neg_alpha[0] = -(alpha[0]);
  }
}
//-----------------------------------------------------------------------------
__global__ void compute_beta(double* rnorm_old, double* rnorm_new, double* beta)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid == 0) {
    beta[0] = (rnorm_new[0] * rnorm_new[0]) / (rnorm_old[0] * rnorm_old[0]);
  }
}
//-----------------------------------------------------------------------------
__global__ void compute_sqrt(double* in, double* out)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid == 0) {
    out[0] = sqrt(in[0]);
  }
}
//-----------------------------------------------------------------------------
std::tuple<double*, int> spmv::cg(MPI_Comm comm, const spmv::Matrix<double>& A,
                                  double* b, int kmax, double rtol)
{
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);

  std::shared_ptr<const spmv::L2GMap> col_l2g = A.col_map();
  std::shared_ptr<const spmv::L2GMap> row_l2g = A.row_map();

  // Check the row map is unghosted
  if (row_l2g->num_ghosts() > 0)
    throw std::runtime_error("spmv::cg - Error: A.row_map() has ghost entries");

  int M = row_l2g->local_size();
  int N_padded = col_l2g->local_size() + col_l2g->num_ghosts();

  // Create CUDA streams to offload operations
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  // Get handle to the CUBLAS context
  cublasHandle_t cublas_handle = 0;
  CHECK_CUBLAS(cublasCreate(&cublas_handle));
  CHECK_CUBLAS(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

  // Allocate device pointers for vectors
  double* d_x = nullptr;
  double* d_p = nullptr;
  double* d_r = nullptr;
  double* d_Ap = nullptr;

  CHECK_CUDA(cudaMalloc((void**)&d_x, N_padded * sizeof(double)));
  CHECK_CUDA(cudaMalloc((void**)&d_p, N_padded * sizeof(double)));
  CHECK_CUDA(cudaMalloc((void**)&d_r, M * sizeof(double)));
  CHECK_CUDA(cudaMalloc((void**)&d_Ap, M * sizeof(double)));
  // FIXME if there are more DMA engines per direction maybe overlap
  // Use asynchronous memory copies to hide launch overheads
  CHECK_CUDA(cudaMemcpyAsync(d_p, b, M * sizeof(double), cudaMemcpyHostToDevice,
                             stream1));
  CHECK_CUDA(cudaMemcpyAsync(d_r, d_p, M * sizeof(double),
                             cudaMemcpyDeviceToDevice, stream1));

  double* d_alpha = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_alpha, sizeof(double)));
  double* d_neg_alpha = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_neg_alpha, sizeof(double)));
  double* d_beta = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_beta, sizeof(double)));

  double* d_rnorm0 = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_rnorm0, sizeof(double)));
  double* d_rnorm_old = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_rnorm_old, sizeof(double)));
  double* d_rnorm_new = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_rnorm_new, sizeof(double)));
  double* d_rnorm_local = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_rnorm_local, sizeof(double)));

  double* d_pdotAp_local = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_pdotAp_local, sizeof(double)));
  double* d_pdotAp = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_pdotAp, sizeof(double)));

  double scalar_one = 1;
  double* d_scalar_one = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_scalar_one, sizeof(double)));
  CHECK_CUDA(cudaMemcpyAsync(d_scalar_one, &scalar_one, sizeof(double),
                             cudaMemcpyHostToDevice, stream1));

  CHECK_CUBLAS(cublasSetStream(cublas_handle, stream1));
  CHECK_CUBLAS(cublasDdot(cublas_handle, M, d_r, 1, d_r, 1, d_rnorm_local));
  cudaStreamSynchronize(stream1);
  MPI_Allreduce(d_rnorm_local, d_rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);
  double rnorm0;
  CHECK_CUDA(
      cudaMemcpy(&rnorm0, d_rnorm0, sizeof(double), cudaMemcpyDeviceToHost));
  rnorm0 = sqrt(rnorm0);
  CHECK_CUDA(
      cudaMemcpy(d_rnorm_old, &rnorm0, sizeof(double), cudaMemcpyHostToDevice));

  // Iterations of CG
  int k = 0;
  cudaEvent_t event;
  cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  while (k < kmax) {
    ++k;

    // Ap = A.p
    col_l2g->update(d_p, stream1);
    A.mult(d_p, d_Ap, stream1);

    // Calculate alpha = r.r/p.Ap
    CHECK_CUBLAS(cublasDdot(cublas_handle, M, d_p, 1, d_Ap, 1, d_pdotAp_local));
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    MPI_Allreduce(d_pdotAp_local, d_pdotAp, 1, MPI_DOUBLE, MPI_SUM, comm);
    compute_alpha<<<1, 1, 0, stream1>>>(d_rnorm_old, d_pdotAp, d_alpha,
                                        d_neg_alpha);
    cudaEventRecord(event, stream1);

    // Update x and r
    // These operations can be done in parallel, so launch in seperate streams
    // x = x + alpha*p
    CHECK_CUDA(cudaStreamWaitEvent(stream2, event));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, stream2));
    CHECK_CUBLAS(cublasDaxpy(cublas_handle, M, d_alpha, d_p, 1, d_x, 1));
    // r = r - alpha*Ap
    CHECK_CUBLAS(cublasSetStream(cublas_handle, stream1));
    CHECK_CUBLAS(cublasDaxpy(cublas_handle, M, d_neg_alpha, d_Ap, 1, d_r, 1));

    // Update rnorm
    CHECK_CUBLAS(cublasDdot(cublas_handle, M, d_r, 1, d_r, 1, d_rnorm_local));
    cudaStreamSynchronize(stream1);
    MPI_Allreduce(d_rnorm_local, d_rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
    compute_sqrt<<<1, 1, 0, stream1>>>(d_rnorm_new, d_rnorm_new);
    compute_beta<<<1, 1, 0, stream1>>>(d_rnorm_old, d_rnorm_new, d_beta);
    // FIXME Can you hide this?
    CHECK_CUDA(cudaMemcpyAsync(d_rnorm_old, d_rnorm_new, sizeof(double),
                               cudaMemcpyDeviceToDevice, stream1));

    double rnorm_new;
    CHECK_CUDA(cudaMemcpyAsync(&rnorm_new, d_rnorm_new, sizeof(double),
                               cudaMemcpyDeviceToHost, stream1));
    cudaStreamSynchronize(stream1);
    if (rnorm_new / rnorm0 < rtol)
      break;

    // Update p.
    // p = r + beta*p
    CHECK_CUBLAS(cublasDscal(cublas_handle, M, d_beta, d_p, 1));
    CHECK_CUBLAS(cublasDaxpy(cublas_handle, M, d_scalar_one, d_r, 1, d_p, 1));
  }

  // Cleanup
  CHECK_CUDA(cudaStreamDestroy(stream1));
  CHECK_CUDA(cudaStreamDestroy(stream2));
  CHECK_CUDA(cudaFree(d_Ap));
  CHECK_CUDA(cudaFree(d_p));
  CHECK_CUDA(cudaFree(d_r));
  CHECK_CUDA(cudaFree(d_scalar_one));
  CHECK_CUDA(cudaFree(d_alpha));
  CHECK_CUDA(cudaFree(d_neg_alpha));
  CHECK_CUDA(cudaFree(d_beta));
  CHECK_CUDA(cudaFree(d_rnorm_old));
  CHECK_CUDA(cudaFree(d_rnorm_new));
  CHECK_CUDA(cudaFree(d_rnorm_local));
  CHECK_CUDA(cudaFree(d_pdotAp_local));
  CHECK_CUDA(cudaFree(d_pdotAp));
  CHECK_CUBLAS(cublasDestroy(cublas_handle));

  return std::make_tuple(d_x, k);
}
//-----------------------------------------------------------------------------

// cudaMallocAsync not supported yet by UCX
// std::tuple<double*, int> spmv::cg(MPI_Comm comm, const spmv::Matrix<double>&
// A, double* b, int kmax, double rtol)
// {
//   int mpi_rank;
//   MPI_Comm_rank(comm, &mpi_rank);

//   std::shared_ptr<const spmv::L2GMap> col_l2g = A.col_map();
//   std::shared_ptr<const spmv::L2GMap> row_l2g = A.row_map();

//   // Check the row map is unghosted
//   if (row_l2g->num_ghosts() > 0)
//     throw std::runtime_error("spmv::cg - Error: A.row_map() has ghost
//     entries");

//   int M = row_l2g->local_size();
//   int N_padded = col_l2g->local_size() + col_l2g->num_ghosts();

//   // Create CUDA stream to offload operations
//   cudaStream_t stream;
//   cudaStreamCreate(&stream);

//   // Get handle to the CUBLAS context
//   cublasHandle_t cublas_handle = 0;
//   CHECK_CUBLAS(cublasCreate(&cublas_handle));
//   CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));
//   CHECK_CUBLAS(cublasSetPointerMode(cublas_handle,
//   CUBLAS_POINTER_MODE_DEVICE));

//   // Allocate device pointers for vectors
//   double* d_x = nullptr;
//   double* d_p = nullptr;
//   double* d_r = nullptr;
//   double* d_Ap = nullptr;
//   CHECK_CUDA(cudaMallocAsync((void **)&d_x, N_padded * sizeof(double),
//   stream)); CHECK_CUDA(cudaMalloc((void **)&d_p, N_padded * sizeof(double)));
//   CHECK_CUDA(cudaMallocAsync((void **)&d_r, M * sizeof(double), stream));
//   CHECK_CUDA(cudaMallocAsync((void **)&d_Ap, M * sizeof(double), stream));
//   CHECK_CUDA(cudaMemcpyAsync(d_p, b, M * sizeof(double),
//   cudaMemcpyHostToDevice, stream)); CHECK_CUDA(cudaMemcpyAsync(d_r, d_p, M *
//   sizeof(double), cudaMemcpyDeviceToDevice, stream));

//   double* d_alpha = nullptr;
//   CHECK_CUDA(cudaMallocAsync((void **)&d_alpha, sizeof(double), stream));
//   double* d_neg_alpha = nullptr;
//   CHECK_CUDA(cudaMallocAsync((void **)&d_neg_alpha, sizeof(double), stream));
//   double* d_beta = nullptr;
//   CHECK_CUDA(cudaMallocAsync((void **)&d_beta, sizeof(double), stream));

//   double* d_rnorm0 = nullptr;
//   CHECK_CUDA(cudaMalloc((void **)&d_rnorm0, sizeof(double)));
//   double* d_rnorm_old = nullptr;
//   CHECK_CUDA(cudaMallocAsync((void **)&d_rnorm_old, sizeof(double), stream));
//   double* d_rnorm_new = nullptr;
//   CHECK_CUDA(cudaMalloc((void **)&d_rnorm_new, sizeof(double)));
//   double* d_rnorm_local = nullptr;
//   CHECK_CUDA(cudaMalloc((void **)&d_rnorm_local, sizeof(double)));

//   double* d_pdotAp_local = nullptr;
//   CHECK_CUDA(cudaMalloc((void **)&d_pdotAp_local, sizeof(double)));
//   double* d_pdotAp = nullptr;
//   CHECK_CUDA(cudaMalloc((void **)&d_pdotAp, sizeof(double)));

//   double scalar_one = 1;
//   double* d_scalar_one = nullptr;
//   CHECK_CUDA(cudaMallocAsync((void **)&d_scalar_one, sizeof(double),
//   stream)); CHECK_CUDA(cudaMemcpyAsync(d_scalar_one, &scalar_one,
//   sizeof(double), cudaMemcpyHostToDevice, stream));

//   int k = 0;
//   CHECK_CUBLAS(cublasDdot(cublas_handle, M, d_r, 1, d_r, 1, d_rnorm_local));
//   cudaStreamSynchronize(stream);
//   MPI_Allreduce(d_rnorm_local, d_rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);
//   double rnorm0;
//   CHECK_CUDA(cudaMemcpy(&rnorm0, d_rnorm0, sizeof(double),
//   cudaMemcpyDeviceToHost)); rnorm0 = sqrt(rnorm0);

//   // Iterations of CG
//   CHECK_CUDA(cudaMemcpy(d_rnorm_old, &rnorm0, sizeof(double),
//   cudaMemcpyHostToDevice)); while (k < kmax) {
//     ++k;

//     // Ap = A.p
//     //col_l2g->update(d_p, stream);
//     A.mult(d_p, d_Ap, stream);

//     // Calculate alpha = r.r/p.Ap
//     CHECK_CUBLAS(cublasDdot(cublas_handle, M, d_p, 1, d_Ap, 1,
//     d_pdotAp_local)); cudaStreamSynchronize(stream);
//     MPI_Allreduce(d_pdotAp_local, d_pdotAp, 1, MPI_DOUBLE, MPI_SUM, comm);
//     compute_alpha<<<1, 1, 0, stream>>>(d_rnorm_old, d_pdotAp, d_alpha,
//     d_neg_alpha);

//     // Update x and r
//     // x = x + alpha*p
//     CHECK_CUBLAS(cublasDaxpy(cublas_handle, M, d_alpha, d_p, 1, d_x, 1));
//     // r = r - alpha*Ap
//     CHECK_CUBLAS(cublasDaxpy(cublas_handle, M, d_neg_alpha, d_Ap, 1, d_r,
//     1));

//     // Update rnorm
//     CHECK_CUBLAS(cublasDdot(cublas_handle, M, d_r, 1, d_r, 1,
//     d_rnorm_local)); cudaStreamSynchronize(stream);
//     MPI_Allreduce(d_rnorm_local, d_rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
//     compute_sqrt<<<1, 1, 0, stream>>>(d_rnorm_new, d_rnorm_new);
//     compute_beta<<<1, 1, 0, stream>>>(d_rnorm_old, d_rnorm_new, d_beta);
//     CHECK_CUDA(cudaMemcpyAsync(d_rnorm_old, d_rnorm_new, sizeof(double),
//     cudaMemcpyDeviceToDevice, stream));

//     double rnorm_new;
//     CHECK_CUDA(cudaMemcpy(&rnorm_new, d_rnorm_new, sizeof(double),
//     cudaMemcpyDeviceToHost)); if (rnorm_new / rnorm0 < rtol)
//       break;

//     // Update p.
//     // p = r + beta*p
//     CHECK_CUBLAS(cublasDscal(cublas_handle, M, d_beta, d_p, 1));
//     CHECK_CUBLAS(cublasDaxpy(cublas_handle, M, d_scalar_one, d_r, 1, d_p,
//     1));
//   }

//   // Cleanup
//   CHECK_CUBLAS(cublasDestroy(cublas_handle));
//   CHECK_CUDA(cudaFreeAsync(d_Ap, stream));
//   CHECK_CUDA(cudaFreeAsync(d_p, stream));
//   CHECK_CUDA(cudaFreeAsync(d_r, stream));
//   CHECK_CUDA(cudaFreeAsync(d_scalar_one, stream));
//   CHECK_CUDA(cudaFreeAsync(d_alpha, stream));
//   CHECK_CUDA(cudaFreeAsync(d_neg_alpha, stream));
//   CHECK_CUDA(cudaFreeAsync(d_beta, stream));
//   CHECK_CUDA(cudaFreeAsync(d_rnorm_old, stream));
//   CHECK_CUDA(cudaFreeAsync(d_rnorm_new, stream));
//   CHECK_CUDA(cudaFreeAsync(d_rnorm_local, stream));
//   CHECK_CUDA(cudaFreeAsync(d_pdotAp_local, stream));
//   CHECK_CUDA(cudaFreeAsync(d_pdotAp, stream));
//   CHECK_CUDA(cudaStreamDestroy(stream));

//   return std::make_tuple(d_x, k);
// }

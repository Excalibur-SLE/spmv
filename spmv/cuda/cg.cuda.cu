// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "cg_cuda.h"

#include "L2GMap.h"
#include "Matrix.h"
#include "cuda_executor.h"
#include "cuda_helper.h"

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
int spmv::cg(MPI_Comm comm, spmv::CudaExecutor& exec,
             const spmv::Matrix<double>& A, const double* b, double* x,
             int kmax, double rtol)
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

  CHECK_CUBLAS(cublasSetPointerMode(exec.get_cublas_handle(),
                                    CUBLAS_POINTER_MODE_DEVICE));

  // Allocate device pointers for vectors
  double* d_x = exec.alloc<double>(N_padded);
  double* d_p = exec.alloc<double>(N_padded);
  double* d_r = exec.alloc<double>(M);
  double* d_Ap = exec.alloc<double>(M);
  exec.copy<double>(d_p, b, M);
  exec.copy<double>(d_r, d_p, M);

  double* d_alpha = exec.alloc<double>(1);
  double* d_neg_alpha = exec.alloc<double>(1);
  double* d_beta = exec.alloc<double>(1);
  double* d_rnorm0 = exec.alloc<double>(1);
  double* d_rnorm_old = exec.alloc<double>(1);
  double* d_rnorm_new = exec.alloc<double>(1);
  double* d_rnorm_local = exec.alloc<double>(1);
  double* d_pdotAp_local = exec.alloc<double>(1);
  double* d_pdotAp = exec.alloc<double>(1);
  double scalar_one = 1;
  double* d_scalar_one = exec.alloc<double>(1);
  exec.copy_from<double>(d_scalar_one, exec.get_host(), &scalar_one, 1);

  CHECK_CUBLAS(cublasSetStream(exec.get_cublas_handle(), stream1));
  CHECK_CUBLAS(
      cublasDdot(exec.get_cublas_handle(), M, d_r, 1, d_r, 1, d_rnorm_local));
  CHECK_CUDA(cudaStreamSynchronize(stream1));
  MPI_Allreduce(d_rnorm_local, d_rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);
  compute_sqrt<<<1, 1, 0, stream1>>>(d_rnorm0, d_rnorm0);
  double rnorm0;
  exec.copy_to<double>(&rnorm0, exec.get_host(), d_rnorm0, 1);
  exec.copy<double>(d_rnorm_old, d_rnorm0, 1);

  // Iterations of CG
  int k = 0;
  cudaEvent_t event;
  cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  while (k < kmax) {
    ++k;

    // Ap = A.p
    exec.set_cuda_stream(stream1);
    col_l2g->update(d_p);
    A.mult(d_p, d_Ap);

    // Calculate alpha = r.r/p.Ap
    CHECK_CUBLAS(cublasDdot(exec.get_cublas_handle(), M, d_p, 1, d_Ap, 1,
                            d_pdotAp_local));
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    MPI_Allreduce(d_pdotAp_local, d_pdotAp, 1, MPI_DOUBLE, MPI_SUM, comm);
    compute_alpha<<<1, 1, 0, stream1>>>(d_rnorm_old, d_pdotAp, d_alpha,
                                        d_neg_alpha);
    cudaEventRecord(event, stream1);

    // Update x and r
    // These operations can be done in parallel, so launch in seperate streams
    // r = r - alpha*Ap
    CHECK_CUBLAS(
        cublasDaxpy(exec.get_cublas_handle(), M, d_neg_alpha, d_Ap, 1, d_r, 1));
    // x = x + alpha*p
    CHECK_CUDA(cudaStreamWaitEvent(stream2, event));
    CHECK_CUBLAS(cublasSetStream(exec.get_cublas_handle(), stream2));
    CHECK_CUBLAS(
        cublasDaxpy(exec.get_cublas_handle(), M, d_alpha, d_p, 1, d_x, 1));

    // Update rnorm
    CHECK_CUBLAS(cublasSetStream(exec.get_cublas_handle(), stream1));
    CHECK_CUBLAS(
        cublasDdot(exec.get_cublas_handle(), M, d_r, 1, d_r, 1, d_rnorm_local));
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    MPI_Allreduce(d_rnorm_local, d_rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
    compute_sqrt<<<1, 1, 0, stream1>>>(d_rnorm_new, d_rnorm_new);
    compute_beta<<<1, 1, 0, stream1>>>(d_rnorm_old, d_rnorm_new, d_beta);
    CHECK_CUDA(cudaMemcpyAsync(d_rnorm_old, d_rnorm_new, sizeof(double),
                               cudaMemcpyDeviceToDevice, stream1));

    double rnorm_new;
    CHECK_CUDA(cudaMemcpyAsync(&rnorm_new, d_rnorm_new, sizeof(double),
                               cudaMemcpyDeviceToHost, stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    if (rnorm_new / rnorm0 < rtol)
      break;

    // Update p.
    // p = r + beta*p
    CHECK_CUBLAS(cublasDscal(exec.get_cublas_handle(), M, d_beta, d_p, 1));
    CHECK_CUBLAS(
        cublasDaxpy(exec.get_cublas_handle(), M, d_scalar_one, d_r, 1, d_p, 1));
  }

  // Copy d_x to x
  exec.copy<double>(x, d_x, M);

  // Cleanup
  exec.free(d_x);
  exec.free(d_Ap);
  exec.free(d_p);
  exec.free(d_r);
  exec.free(d_scalar_one);
  exec.free(d_alpha);
  exec.free(d_neg_alpha);
  exec.free(d_beta);
  exec.free(d_rnorm0);
  exec.free(d_rnorm_old);
  exec.free(d_rnorm_new);
  exec.free(d_rnorm_local);
  exec.free(d_pdotAp_local);
  exec.free(d_pdotAp);
  exec.reset_cuda_stream();
  CHECK_CUDA(cudaStreamDestroy(stream1));
  CHECK_CUDA(cudaStreamDestroy(stream2));

  return k;
}
//-----------------------------------------------------------------------------

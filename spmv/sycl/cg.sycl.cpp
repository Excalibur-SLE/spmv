// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "cg_sycl.h"

#include "L2GMap.h"
#include "Matrix.h"
#include "sycl_executor.h"

#ifdef _DPCPP
#include "oneapi/mkl.hpp"
#else
#include "blas_sycl.h"
#endif

//-----------------------------------------------------------------------------
int spmv::cg(MPI_Comm comm, SyclExecutor& exec, const spmv::Matrix<double>& A,
             const double* b, double* x, int kmax, double rtol)
{
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);

  sycl::queue* q = exec.get_queue();

  std::shared_ptr<const spmv::L2GMap> col_l2g = A.col_map();
  std::shared_ptr<const spmv::L2GMap> row_l2g = A.row_map();

  // Check the row map is unghosted
  if (row_l2g->num_ghosts() > 0)
    throw std::runtime_error("spmv::cg - Error: A.row_map() has ghost entries");

  int M = row_l2g->local_size();
  int N_padded = col_l2g->local_size() + col_l2g->num_ghosts();

  // Allocate vectors
  double* x_padded = exec.alloc<double>(N_padded);
  double* p = exec.alloc<double>(N_padded);
  double* r = exec.alloc<double>(M);
  double* Ap = exec.alloc<double>(M);

  exec.copy<double>(p, b, M);
  exec.copy<double>(r, b, M);

  double rnorm;
#ifdef _DPCPP
  oneapi::mkl::blas::row_major::dot(*q, M, r, 1, r, 1, &rnorm).wait();
#else
  squared_norm(M, r, &rnorm, *q).wait();
#endif
  double rnorm0;
  MPI_Allreduce(&rnorm, &rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);
  rnorm0 = std::sqrt(rnorm0);

  // Iterations of CG
  int k = 0;
  //  const double rtol2 = rtol * rtol;
  double rnorm_old = rnorm0;
  while (k < kmax) {
    ++k;

    // Ap = A.p
    col_l2g->update(p);
    A.mult(p, Ap);

    // Calculate alpha = r.r/p.y
    double pdotAp_local;
#ifdef _DPCPP
    oneapi::mkl::blas::row_major::dot(*q, M, p, 1, Ap, 1, &pdotAp_local).wait();
#else
    dot(M, p, Ap, &pdotAp_local, *q).wait();
#endif
    double pdotAp;
    MPI_Allreduce(&pdotAp_local, &pdotAp, 1, MPI_DOUBLE, MPI_SUM, comm);
    double alpha = (rnorm_old * rnorm_old) / pdotAp;

    // Update x and r
#ifdef _DPCPP
    oneapi::mkl::blas::row_major::axpy(*q, M, alpha, p, 1, x_padded, 1).wait();
    oneapi::mkl::blas::row_major::axpy(*q, M, -alpha, Ap, 1, r, 1).wait();
#else
    fused_update(M, alpha, p, x_padded, Ap, r, *q).wait();
#endif

    // Update rnorm
#ifdef _DPCPP
    oneapi::mkl::blas::row_major::dot(*q, M, r, 1, r, 1, &rnorm).wait();
#else
    squared_norm(M, r, &rnorm, *q).wait();
#endif
    double rnorm_new;
    MPI_Allreduce(&rnorm, &rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
    rnorm_new = std::sqrt(rnorm_new);
    double beta = (rnorm_new * rnorm_new) / (rnorm_old * rnorm_old);
    rnorm_old = rnorm_new;

    if (rnorm_new / rnorm0 < rtol)
      break;

    // Update p.
    // p = r + beta*p
#ifdef _DPCPP
    oneapi::mkl::blas::row_major::scal(*q, M, beta, p, 1).wait();
    oneapi::mkl::blas::row_major::axpy(*q, M, 1, r, 1, p, 1).wait();
#else
    axpy(M, beta, p, r, p, *q).wait();
#endif
  }

  // Copy x_padded to x
  exec.copy<double>(x, x_padded, M);
  exec.free(x_padded);
  exec.free(p);
  exec.free(r);
  exec.free(Ap);

  return k;
}
//-----------------------------------------------------------------------------

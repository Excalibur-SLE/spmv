// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "cg_openmp.h"

#include "L2GMap.h"
#include "Matrix.h"
#include "omp_executor.h"

#include <iomanip>
#include <iostream>
#include <memory>

#ifdef _BLAS_MKL
#include <mkl.h>
#endif
#ifdef _BLAS_OPENBLAS
#include <cblas.h>
#endif

//-----------------------------------------------------------------------------
int spmv::cg(MPI_Comm comm, spmv::OmpExecutor& exec,
             const spmv::Matrix<double>& A, double* b, double* x, int kmax,
             double rtol)
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

  // Allocate device pointers for vectors
  double* r = exec.alloc<double>(M);
  double* Ap = exec.alloc<double>(M);
  double* x_padded = exec.alloc<double>(N_padded);
  double* p = exec.alloc<double>(N_padded);

  exec.copy<double>(r, b, M);
  exec.copy<double>(p, b, M);

  double rnorm = cblas_ddot(M, r, 1, r, 1);
  double rnorm0;
  MPI_Allreduce(&rnorm, &rnorm0, 1, mpi_type<double>(), MPI_SUM, comm);
  rnorm0 = std::sqrt(rnorm0);

  // Iterations of CG
  double rnorm_old = rnorm0;
  int k = 0;
  while (k < kmax) {
    ++k;

    // Ap = A.p
    col_l2g->update(p);
    A.mult(p, Ap);

    // Calculate alpha = r.r/p.Ap
    double pdotAp_local = cblas_ddot(M, p, 1, Ap, 1);
    double pdotAp;
    MPI_Allreduce(&pdotAp_local, &pdotAp, 1, mpi_type<double>(), MPI_SUM, comm);
    double alpha = (rnorm_old * rnorm_old) / pdotAp;

    // Update x and r
    cblas_daxpy(M, alpha, p, 1, x_padded, 1);
    cblas_daxpy(M, -alpha, Ap, 1, r, 1);

    // Update rnorm
    rnorm = cblas_ddot(M, r, 1, r, 1);
    double rnorm_new;
    MPI_Allreduce(&rnorm, &rnorm_new, 1, mpi_type<double>(), MPI_SUM, comm);
    rnorm_new = std::sqrt(rnorm_new);
    double beta = (rnorm_new * rnorm_new) / (rnorm_old * rnorm_old);
    rnorm_old = rnorm_new;

    if (rnorm_new / rnorm0 < rtol)
      break;

    // Update p
    cblas_dscal(M, beta, p, 1);
    cblas_daxpy(M, 1, r, 1, p, 1);
  }

  // Copy x_padded to x
  exec.copy<double>(x, x_padded, M);

  // Cleanup
  exec.free(r);
  exec.free(Ap);
  exec.free(x_padded);
  exec.free(p);

  return k;
}
//-----------------------------------------------------------------------------

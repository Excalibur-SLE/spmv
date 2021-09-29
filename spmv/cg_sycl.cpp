// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "cg_sycl.h"
#include "L2GMap.h"
#include "Matrix.h"
#include <iomanip>

//-----------------------------------------------------------------------------
std::tuple<double*, int> spmv::cg(MPI_Comm comm, sycl::queue& q,
                                  const spmv::Matrix<double>& A, double* b,
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

  // if (b.rows() != M)
  //   throw std::runtime_error("spmv::cg - Error: b.rows() != A.rows()");

  // Allocate auxiliary vectors
  auto r = sycl::malloc_shared<double>(M, q);
  auto y = sycl::malloc_shared<double>(M, q);
  auto p = sycl::malloc_shared<double>(
      col_l2g->local_size() + col_l2g->num_ghosts(), q);
  auto x = sycl::malloc_shared<double>(
      col_l2g->local_size() + col_l2g->num_ghosts(), q);

  // prefetch + mem_advise
  q.memset(x, 0, M * sizeof(double));
  q.memcpy(r, b, M * sizeof(double));
  q.memcpy(p, b, M * sizeof(double));
  q.wait();

  const double rnorm0_local = squared_norm(q, M, r);
  double rnorm0;
  MPI_Allreduce(&rnorm0_local, &rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);

  //  double rnorm_old = rnorm0;
  // Iterations of CG
  const double rtol2 = rtol * rtol;
  double rnorm = rnorm0;
  int k = 0;
  while (k < kmax)
  {
    ++k;

    // y = A.p
    col_l2g->update(p);
    A.mult(q, p, y);

    // Calculate alpha = r.r/p.y
    const double pdoty_local = dot(q, M, p, y);
    double pdoty;
    MPI_Allreduce(&pdoty_local, &pdoty, 1, MPI_DOUBLE, MPI_SUM, comm);
    const double alpha = rnorm / pdoty;

    // // Update x and r
    // // x = x + alpha*p
    // axpy(q, M, alpha, p, x, x);

    // // r = r - alpha*y
    // axpy(q, M, -alpha, y, r, r);
    // q.wait();

    fused_update(q, M, alpha, p, x, y, r);
    q.wait();

    // Update rnorm
    const double rnorm_new_local = squared_norm(q, M, r);
    double rnorm_new;
    MPI_Allreduce(&rnorm_new_local, &rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
    const double beta = rnorm_new / rnorm;
    rnorm = rnorm_new;

    if (rnorm / rnorm0 < rtol2)
      break;

    // Update p.
    // p = r + beta*p
    axpy(q, M, beta, p, r, p).wait();
  }

  sycl::free(r, q);
  sycl::free(y, q);
  sycl::free(p, q);
  // sycl::free(x, q);

  return std::make_tuple(x, k);
}
//-----------------------------------------------------------------------------

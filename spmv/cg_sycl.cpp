// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "cg_sycl.h"
#include "L2GMap.h"
#include "Matrix.h"

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
  int N_padded = col_l2g->local_size() + col_l2g->num_ghosts();

  // Allocate auxiliary vectors
  auto r = sycl::malloc_shared<double>(M, q);
  // This buffer is reduced by the host during the SpMV operation so it needs to
  // be shared between host and device
  auto y = sycl::malloc_shared<double>(M, q);
  // This buffer is used by MPI so it needs to be shared between host and device
  auto p = sycl::malloc_shared<double>(N_padded, q);
  // This buffer is returned to the user so it needs to be shared between host
  // and device
  auto x = sycl::malloc_shared<double>(N_padded, q);

  q.prefetch(r, M * sizeof(double));
  q.prefetch(y, M * sizeof(double));
  q.prefetch(p, N_padded * sizeof(double));
  q.prefetch(x, N_padded * sizeof(double));

  // Initialise vectors
  q.memset(x, 0, N_padded * sizeof(double));
  q.memcpy(r, b, M * sizeof(double)).wait_and_throw(); // b - A * x0
  q.memcpy(p, b, M * sizeof(double)).wait_and_throw();

  sycl::event event;
  double rnorm;
  squared_norm(q, M, r, &rnorm).wait_and_throw();
  double rnorm0;
  MPI_Allreduce(&rnorm, &rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);

  // Iterations of CG
  const double rtol2 = rtol * rtol;
  double rnorm_old = rnorm0;
  int k = 0;
  while (k < kmax)
  {
    ++k;

    // y = A.p
    col_l2g->update(p);
    event = A.mult(q, p, y);

    // Calculate alpha = r.r/p.y
    double pdoty_local;
    dot(q, M, p, y, &pdoty_local, {event}).wait_and_throw();
    double pdoty;
    MPI_Allreduce(&pdoty_local, &pdoty, 1, MPI_DOUBLE, MPI_SUM, comm);
    const double alpha = rnorm_old / pdoty;

    // Update x and r
    // x = x + alpha*p
    // axpy(q, M, alpha, p, x, x);
    // // r = r - alpha*y x y r
    // axpy(q, M, -alpha, y, r, r);
    // q.wait();
    event = fused_update(q, M, alpha, p, x, y, r);

    // Update rnorm
    squared_norm(q, M, r, &rnorm, {event}).wait_and_throw();
    double rnorm_new;
    MPI_Allreduce(&rnorm, &rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
    double beta = rnorm_new / rnorm_old;
    rnorm_old = rnorm_new;

    // Update p.
    // p = r + beta*p    x y r
    axpy(q, M, beta, p, r, p).wait_and_throw();

    if (rnorm_new / rnorm0 < rtol2)
      break;
  }

  sycl::free(r, q);
  sycl::free(y, q);
  sycl::free(p, q);

  return std::make_tuple(std::move(x), k);
}
//-----------------------------------------------------------------------------

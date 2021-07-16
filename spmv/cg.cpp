// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// SPDX-License-Identifier:    MIT

#include "cg.h"
#include "L2GMap.h"
#include "Matrix.h"
#include <iomanip>
#include <iostream>

//-----------------------------------------------------------------------------
std::tuple<Eigen::VectorXd, int>
spmv::cg(MPI_Comm comm, const spmv::Matrix<double>& A,
         const Eigen::Ref<const Eigen::VectorXd>& b, int kmax, double rtol)
{
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);

  std::shared_ptr<const spmv::L2GMap> col_l2g = A.col_map();
  std::shared_ptr<const spmv::L2GMap> row_l2g = A.row_map();

  // Check the row map is unghosted
  if (row_l2g->num_ghosts() > 0)
    throw std::runtime_error("spmv::cg - Error: A.row_map() has ghost entries");

  int M = row_l2g->local_size();

  if (b.rows() != M)
    throw std::runtime_error("spmv::cg - Error: b.rows() != A.rows()");

  // Residual vector
  Eigen::VectorXd r(M);
  Eigen::VectorXd y(M);
  Eigen::VectorXd x(col_l2g->local_size() + col_l2g->num_ghosts());
  Eigen::VectorXd p(col_l2g->local_size() + col_l2g->num_ghosts());
  p.setZero();

  // Assign to dense part of sparse vector
  x.setZero();
  r = b; // b - A * x0
  p.head(M) = r;

  double rnorm = r.squaredNorm();
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
    col_l2g->update(p.data());
    y = A * p;

    // Calculate alpha = r.r/p.y
    double pdoty = p.head(M).dot(y);
    double pdoty_sum;
    MPI_Allreduce(&pdoty, &pdoty_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    double alpha = rnorm_old / pdoty_sum;

    // Update x and r
    x.head(M) += alpha * p.head(M);
    r -= alpha * y;

    // Update rnorm
    rnorm = r.squaredNorm();
    double rnorm_new;
    MPI_Allreduce(&rnorm, &rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
    double beta = rnorm_new / rnorm_old;
    rnorm_old = rnorm_new;

    // Update p
    p.head(M) = p.head(M) * beta + r;

    if (rnorm_new / rnorm0 < rtol2)
      break;
  }

  return std::make_tuple(std::move(x), k);
}
//---------------------
#ifdef _SYCL
std::tuple<double*, int> spmv::cg_sycl(MPI_Comm comm, sycl::queue& q,
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
    A.spmv_sycl(q, p, y).wait();
    // A.spmv_sym_sycl(q, p, y).wait();

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
#endif
//-----------------------------------------------------------------------------

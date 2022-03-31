// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "L2GMap.h"
#include "Matrix.h"
#include "blas_sycl.h"
#include "cg_sycl.h"

//-----------------------------------------------------------------------------
std::tuple<double*, int> spmv::cg(MPI_Comm comm, const spmv::Matrix<double>& A,
                                  sycl::buffer<double>& b_buf, int kmax,
                                  double rtol, sycl::queue& q)
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

  // Allocate output vector
  double* x = new double[N_padded]();

  int k = 0;
  {
    // Define SYCL buffers
    sycl::buffer<double> x_buf{x, sycl::range(N_padded)};
    sycl::buffer<double> p_buf{sycl::range(N_padded)};
    sycl::buffer<double> r_buf{sycl::range(M)};
    sycl::buffer<double> Ap_buf{sycl::range(M)};

    q.submit([&](cl::sycl::handler& cgh) {
      auto src_b = b_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto dest_p
          = p_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.copy(src_b, dest_p);
    });
    q.submit([&](cl::sycl::handler& cgh) {
       auto src_b = b_buf.get_access<cl::sycl::access::mode::read>(cgh);
       auto dest_r
           = r_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
       cgh.copy(src_b, dest_r);
     }).wait();

    double rnorm;
    squared_norm(M, r_buf, &rnorm, q);
    double rnorm0;
    MPI_Allreduce(&rnorm, &rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);

    // Iterations of CG
    const double rtol2 = rtol * rtol;
    double rnorm_old = rnorm0;
    while (k < kmax) {
      ++k;

      // Ap = A.p
      {
        sycl::host_accessor p{p_buf, sycl::read_write};
        col_l2g->update(p.get_pointer());
      }
      A.mult(p_buf, Ap_buf, q);

      // Calculate alpha = r.r/p.y
      double pdotAp_local;
      dot(M, p_buf, Ap_buf, &pdotAp_local, q);
      double pdotAp;
      MPI_Allreduce(&pdotAp_local, &pdotAp, 1, MPI_DOUBLE, MPI_SUM, comm);
      double alpha = rnorm_old / pdotAp;

      // Update x and r
      // x = x + alpha*p
      // axpy(M, alpha, p, x, x, q);
      // // r = r - alpha*y x y r
      // axpy(M, -alpha, y, r, r, q);
      fused_update(M, alpha, p_buf, x_buf, Ap_buf, r_buf, q);

      // Update rnorm
      squared_norm(M, r_buf, &rnorm, q);
      double rnorm_new;
      MPI_Allreduce(&rnorm, &rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
      double beta = rnorm_new / rnorm_old;
      rnorm_old = rnorm_new;

      if (rnorm_new / rnorm0 < rtol2)
        break;

      // Update p.
      // p = r + beta*p
      axpy(M, beta, p_buf, r_buf, p_buf, q);
    }
  } // SYCL buffer destruction point

  return std::make_tuple(x, k);
}
//-----------------------------------------------------------------------------
std::tuple<double*, int> spmv::cg(MPI_Comm comm, const spmv::Matrix<double>& A,
                                  double* b, int kmax, double rtol,
                                  sycl::queue& q)
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

  // Allocate vectors
  double* x = sycl::malloc_shared<double>(N_padded, q);
  double* p = sycl::malloc_shared<double>(N_padded, q);
  double* r = sycl::malloc_shared<double>(M, q);
  double* Ap = sycl::malloc_shared<double>(M, q);
  q.memset(x, 0, N_padded * sizeof(double));
  q.memset(p, 0, N_padded * sizeof(double));
  q.memset(r, 0, M * sizeof(double));
  q.wait();
  q.memcpy(p, b, M * sizeof(double));
  q.memcpy(r, b, M * sizeof(double));
  q.wait();

  int k = 0;
  double rnorm;
  squared_norm(M, r, &rnorm, q).wait();
  double rnorm0;
  MPI_Allreduce(&rnorm, &rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);

  // Iterations of CG
  const double rtol2 = rtol * rtol;
  double rnorm_old = rnorm0;
  while (k < kmax) {
    ++k;

    // Ap = A.p
    col_l2g->update(p);
    A.mult(p, Ap, q).wait();

    // Calculate alpha = r.r/p.y
    double pdotAp_local;
    dot(M, p, Ap, &pdotAp_local, q).wait();
    double pdotAp;
    MPI_Allreduce(&pdotAp_local, &pdotAp, 1, MPI_DOUBLE, MPI_SUM, comm);
    double alpha = rnorm_old / pdotAp;

    // Update x and r
    // x = x + alpha*p
    // axpy(M, alpha, p, x, x, q);
    // // r = r - alpha*y x y r
    // axpy(M, -alpha, y, r, r, q);
    // q.wait();
    fused_update(M, alpha, p, x, Ap, r, q).wait();

    // Update rnorm
    squared_norm(M, r, &rnorm, q).wait();
    double rnorm_new;
    MPI_Allreduce(&rnorm, &rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
    double beta = rnorm_new / rnorm_old;
    rnorm_old = rnorm_new;

    if (rnorm_new / rnorm0 < rtol2)
      break;

    // Update p.
    // p = r + beta*p
    axpy(M, beta, p, r, p, q).wait();
  }

  sycl::free(p, q);
  sycl::free(r, q);
  sycl::free(Ap, q);
  q.wait();

  return std::make_tuple(x, k);
}
//-----------------------------------------------------------------------------

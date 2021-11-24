// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "L2GMap.h"
#include "Matrix.h"
#include "cg_sycl.h"

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

  // Allocate output vector
  double* x = new double[N_padded]();
  double* p = new double[N_padded]();
  double* r = new double[M]();
  memcpy(p, b, M * sizeof(double));
  memcpy(r, b, M * sizeof(double));

  int k = 0;
  {
    // Define SYCL buffers
    sycl::buffer<double> x_buf{x, sycl::range(N_padded)};
    sycl::buffer<double> p_buf{p, sycl::range(N_padded)};
    sycl::buffer<double> r_buf{r, sycl::range(M)};
    sycl::buffer<double> Ap_buf{sycl::range(M)};

    double rnorm;
    squared_norm(q, M, r_buf, &rnorm);

    double rnorm0;
    MPI_Allreduce(&rnorm, &rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);
    rnorm0 = sqrt(rnorm0);

    // Iterations of CG
    double rnorm_old = rnorm0;
    while (k < kmax) {
      ++k;

      // Ap = A.p
      {
        sycl::host_accessor p{p_buf, sycl::read_write};
        col_l2g->update(p.get_pointer());
      }

      A.mult(q, p_buf, Ap_buf);

      // Calculate alpha = r.r/p.y
      double pdotAp_local;
      dot(q, M, p_buf, Ap_buf, &pdotAp_local);

      double pdotAp;
      MPI_Allreduce(&pdotAp_local, &pdotAp, 1, MPI_DOUBLE, MPI_SUM, comm);
      const double alpha = (rnorm_old * rnorm_old) / pdotAp;

      // Update x and r
      // x = x + alpha*p
      // axpy(q, M, alpha, p, x, x);
      // // r = r - alpha*y x y r
      // axpy(q, M, -alpha, y, r, r);
      // q.wait();
      fused_update(q, M, alpha, p_buf, x_buf, Ap_buf, r_buf);

      // Update rnorm
      squared_norm(q, M, r_buf, &rnorm);
      double rnorm_new;
      MPI_Allreduce(&rnorm, &rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
      rnorm_new = sqrt(rnorm_new);
      double beta = rnorm_new / rnorm_old;
      rnorm_old = rnorm_new;

      if (rnorm_new / rnorm0 < rtol)
        break;

      // Update p.
      // p = r + beta*p
      axpy(q, M, beta, p_buf, r_buf, p_buf);
    }
  } // SYCL buffer destruction point

  delete[] p;
  delete[] r;

  return std::make_tuple(x, k);
}
//-----------------------------------------------------------------------------

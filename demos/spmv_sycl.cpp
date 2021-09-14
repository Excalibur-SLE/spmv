// Copyright (C) 2018-2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>

#include "CreateA.h"
#include <spmv/L2GMap.h>
#include <spmv/Matrix.h>
#include <spmv/read_petsc.h>

void matvec_main()
{
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  sycl::queue queue(sycl::cpu_selector{});

  // Keep list of timings
  std::map<std::string, std::chrono::duration<double>> timings;

  auto timer_start = std::chrono::system_clock::now();

  // Either create a simple 1D stencil
  spmv::Matrix<double> A = create_A(MPI_COMM_WORLD, 2000000);

  // Or read file created with "-ksp_view_mat binary" option
  //  spmv::Matrix A = spmv::read_petsc_binary(MPI_COMM_WORLD, "A4.dat");

  std::shared_ptr<const spmv::L2GMap> l2g = A.col_map();

  // Get local and global sizes
  std::int64_t M = A.row_map()->local_size();
  std::int64_t N = l2g->global_size();
  // Eigen::VectorXd b(M);
  // auto b = sycl::malloc_shared<double>(M, queue);

  auto timer_end = std::chrono::system_clock::now();
  //    timings["0.MatCreate"] += (timer_end - timer_start);

  timer_start = std::chrono::system_clock::now();

  if (mpi_rank == 0)
    std::cout << "Creating vector of size " << N << "\n";

  // Vector with extra space for ghosts at end
  // Eigen::VectorXd psp(l2g->local_size() + l2g->num_ghosts());
  auto psp = sycl::malloc_shared<double>(l2g->local_size() + l2g->num_ghosts(),
                                         queue);

  // Set up values in local range
  int r0 = l2g->global_offset();
  for (int i = 0; i < M; ++i)
  {
    double z = (double)(i + r0) / double(N);
    psp[i] = exp(-10 * pow(5 * (z - 0.5), 2.0));
  }

  timer_end = std::chrono::system_clock::now();
  timings["1.VecCreate"] += (timer_end - timer_start);

  // Apply matrix a few times
  int n_apply = 1000;
  if (mpi_rank == 0)
    std::cout << "Applying matrix " << n_apply << " times\n";

  // Temporary variable
  // Eigen::VectorXd q(M);
  auto q = sycl::malloc_shared<double>(M, queue);
  for (int i = 0; i < n_apply; ++i)
  {
    timer_start = std::chrono::system_clock::now();
    l2g->update(psp);
    timer_end = std::chrono::system_clock::now();
    timings["2.SparseUpdate"] += (timer_end - timer_start);

    timer_start = std::chrono::system_clock::now();
    A.spmv_sycl(queue, psp, q);
    timer_end = std::chrono::system_clock::now();
    timings["3.SpMV"] += (timer_end - timer_start);

    // timer_start = std::chrono::system_clock::now();
    // psp = q;
    // timer_end = std::chrono::system_clock::now();
    // timings["4.Copy"] += (timer_end - timer_start);
  }

  // double pnorm = psp.head(M).squaredNorm();
  // double pnorm_sum;
  // MPI_Allreduce(&pnorm, &pnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  if (mpi_rank == 0)
    std::cout << "\nTimings (" << mpi_size
              << ")\n----------------------------\n";

  std::chrono::duration<double> total_time
      = std::chrono::duration<double>::zero();
  for (auto q : timings)
    total_time += q.second;
  timings["Total"] = total_time;

  for (auto q : timings)
  {
    double q_local = q.second.count(), q_max, q_min;
    MPI_Reduce(&q_local, &q_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&q_local, &q_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
      std::string pad(16 - q.first.size(), ' ');
      std::cout << "[" << q.first << "]" << pad << q_min << '\t' << q_max
                << "\n";
    }
  }

  // if (mpi_rank == 0)
  // {
  //   std::cout << "----------------------------\n";
  //   std::cout << "norm = " << pnorm_sum << "\n";
  // }
}

//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
#ifdef _OPENMP
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED)
  {
    std::cout << "The threading support level is lesser than required"
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
#else
  MPI_Init(&argc, &argv);
#endif

  matvec_main();

  MPI_Finalize();
  return 0;
}

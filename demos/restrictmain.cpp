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
#include <spmv/read_petsc.h>

void restrict_main()
{
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Keep list of timings
  std::map<std::string, std::chrono::duration<double>> timings;

  auto timer_start = std::chrono::system_clock::now();
  // Read in a PETSc binary format matrix
  auto R = spmv::read_petsc_binary_matrix(MPI_COMM_WORLD, "R4.dat");
  auto q = spmv::read_petsc_binary_vector(MPI_COMM_WORLD, "b4.dat");

  // Get local and global sizes
  std::int64_t M = R.rows();
  auto l2g = R.col_map();
  std::int64_t N = l2g->global_size();

  std::cout << "Vector = " << q.size() << " " << M << "\n";

  auto timer_end = std::chrono::system_clock::now();
  timings["0.PetscRead"] += (timer_end - timer_start);

  timer_start = std::chrono::system_clock::now();

  if (mpi_rank == 0)
    std::cout << "Creating vector of size " << N << "\n";

  // Vector in "column space" with extra space for ghosts at end
  Eigen::VectorXd psp(l2g->local_size() + l2g->num_ghosts());

  timer_end = std::chrono::system_clock::now();
  timings["1.VecCreate"] += (timer_end - timer_start);

  // Apply matrix
  if (mpi_rank == 0)
    std::cout << "Applying matrix\n";

  double pnorm_sum, qnorm_sum;
  for (int i = 0; i < 10; ++i)
  {
    // Restrict
    timer_start = std::chrono::system_clock::now();
    psp = R.transpmult(q);

    timer_end = std::chrono::system_clock::now();
    timings["3.SpMV"] += (timer_end - timer_start);

    timer_start = std::chrono::system_clock::now();
    l2g->reverse_update(psp.data());
    timer_end = std::chrono::system_clock::now();
    timings["2.SparseUpdate"] += (timer_end - timer_start);

    Eigen::Map<Eigen::VectorXd> p(psp.data(), l2g->local_size());
    double pnorm = p.squaredNorm();
    MPI_Allreduce(&pnorm, &pnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Prolongate
    timer_start = std::chrono::system_clock::now();
    l2g->update(psp.data());
    timer_end = std::chrono::system_clock::now();
    timings["2.SparseUpdate"] += (timer_end - timer_start);

    timer_start = std::chrono::system_clock::now();
    q = R.mult(psp);

    timer_end = std::chrono::system_clock::now();
    timings["3.SpMV"] += (timer_end - timer_start);

    double qnorm = q.squaredNorm();
    MPI_Allreduce(&qnorm, &qnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

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

  if (mpi_rank == 0)
  {
    std::cout << "----------------------------\n";
    std::cout << "norm q = " << qnorm_sum << "\n";
    std::cout << "norm p = " << pnorm_sum << "\n";
  }
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

  restrict_main();

  MPI_Finalize();
  return 0;
}

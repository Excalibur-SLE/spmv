// Copyright (C) 2018-2020 Chris Richardson (chris@bpi.cam.ac.uk)
// Copyright (C) 2021-2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <chrono>
#include <iostream>
#include <memory>

#include <mpi.h>

#include <spmv/spmv.h>
#ifdef _BLAS_MKL
#include <mkl.h>
#endif
#ifdef _BLAS_OPENBLAS
#include <cblas.h>
#endif

void cg_main(int argc, char** argv)
{
  // Turn off profiling
  MPI_Pcontrol(0);

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Keep list of timings
  std::map<std::string, std::chrono::duration<double>> timings;

  auto timer_start = std::chrono::system_clock::now();

  std::string argv1, argv2;
  if (argc == 3) {
    argv1 = argv[1];
    argv2 = argv[2];
  } else {
    throw std::runtime_error("Use: ./demo_cg <matrix_file> <vector_file>");
  }

  // Read matrix
  bool symmetric = false;
  spmv::CommunicationModel cm = spmv::CommunicationModel::collective_blocking;
  std::shared_ptr<spmv::ReferenceExecutor> exec
      = spmv::ReferenceExecutor::create();
  auto A = spmv::read_petsc_binary_matrix(argv1, MPI_COMM_WORLD, exec,
                                          symmetric, cm);

  // Read vector
  auto b = spmv::read_petsc_binary_vector(MPI_COMM_WORLD, exec.get(), argv2);
  double* x = exec->alloc<double>(A.rows());

  // Get local and global sizes
  std::shared_ptr<const spmv::L2GMap> l2g = A.col_map();
  std::int64_t N = l2g->global_size();

  if (mpi_rank == 0)
    std::cout << "Global vec size = " << N << "\n";

  auto timer_end = std::chrono::system_clock::now();
  timings["0.ReadPetsc"] += (timer_end - timer_start);

  int max_its = 100;
  double rtol = 1e-10;

  // Turn on profiling for solver only
  MPI_Pcontrol(1);
  timer_start = std::chrono::system_clock::now();
  int num_its = spmv::cg(MPI_COMM_WORLD, *exec, A, b, x, max_its, rtol);
  timer_end = std::chrono::system_clock::now();
  timings["1.Solve"] += (timer_end - timer_start);
  MPI_Pcontrol(0);

  // Test result
  std::int64_t M = l2g->local_size();
  std::int64_t N_local_padded = l2g->local_size() + l2g->num_ghosts();
  double* x_padded = exec->alloc<double>(N_local_padded);
  exec->copy<double>(x_padded, x, M);
  l2g->update(x_padded);
  double* r = exec->alloc<double>(M);
  A.mult(x_padded, r);
  cblas_daxpy(M, -1, b, 1, r, 1);
  double rnorm = cblas_ddot(M, r, 1, r, 1);
  double rnorm_sum;
  MPI_Allreduce(&rnorm, &rnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Get norm of solution vector
  double xnorm = cblas_ddot(M, x, 1, x, 1);
  double xnorm_sum;
  MPI_Allreduce(&xnorm, &xnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    std::cout << "r.norm = " << std::sqrt(rnorm_sum) << "\n";
    std::cout << "x.norm = " << std::sqrt(xnorm_sum) << " in " << num_its
              << " iterations\n";
    std::cout << "\nTimings (" << mpi_size
              << ")\n----------------------------\n";
  }

  std::chrono::duration<double> total_time
      = std::chrono::duration<double>::zero();
  for (auto q : timings)
    total_time += q.second;
  timings["Total"] = total_time;

  for (auto q : timings) {
    double q_local = q.second.count(), q_max, q_min;
    MPI_Reduce(&q_local, &q_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&q_local, &q_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
      std::string pad(16 - q.first.size(), ' ');
      std::cout << "[" << q.first << "]" << pad << q_min << '\t' << q_max
                << "\n";
    }
  }

  if (mpi_rank == 0)
    std::cout << "----------------------------\n";

  // Cleanup
  exec->free(x);
  exec->free(x_padded);
  exec->free(r);
  exec->free(b);
}
//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  cg_main(argc, argv);

  MPI_Finalize();
  return 0;
}

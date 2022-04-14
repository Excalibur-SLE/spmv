// Copyright (C) 2018-2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>

#include <spmv/spmv.h>

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
    throw std::runtime_error("Use: ./cg_demo <matrix_file> <vector_file>");
  }

  // Read matrix
  bool symmetric = false;
  spmv::CommunicationModel cm = spmv::CommunicationModel::collective_blocking;
  std::shared_ptr<spmv::DeviceExecutor> exec
      = spmv::OmpExecutor::create();
  auto A = spmv::read_petsc_binary_matrix(argv1, MPI_COMM_WORLD, exec,
                                          symmetric, cm);

  // Read vector
  auto b = spmv::read_petsc_binary_vector(MPI_COMM_WORLD, argv2);

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
  auto [x, num_its] = spmv::cg(MPI_COMM_WORLD, A, b, max_its, rtol);
  timer_end = std::chrono::system_clock::now();
  timings["1.Solve"] += (timer_end - timer_start);
  MPI_Pcontrol(0);

  // Test result
  l2g->update(x.data());
  Eigen::VectorXd r = A.mult(x) - b;
  double rnorm = r.squaredNorm();
  double rnorm_sum;
  MPI_Allreduce(&rnorm, &rnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Get norm of solution vector
  double xnorm = x.head(l2g->local_size()).squaredNorm();
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
}
//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
#ifdef _OPENMP
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED) {
    std::cout << "The threading support level is lesser than required"
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
#else
  MPI_Init(&argc, &argv);
#endif

  cg_main(argc, argv);

  MPI_Finalize();
  return 0;
}

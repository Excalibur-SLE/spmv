// Copyright (C) 2018-2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>

#include <spmv/spmv.h>
#include <spmv/sycl/blas_sycl.h>

void spmv_main(int argc, char** argv)
{
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::string argv1;
  if (argc == 2) {
    argv1 = argv[1];
  } else {
    throw std::runtime_error("Use: ./spmv_demo <matrix_file>");
  }

  sycl::queue queue(sycl::default_selector{});

  // Keep list of timings
  std::map<std::string, std::chrono::duration<double>> timings;

  auto timer_start = std::chrono::system_clock::now();
  // Read matrix from file created with "-ksp_view_mat binary" option
  bool symmetric = false;
  spmv::CommunicationModel cm = spmv::CommunicationModel::p2p_blocking;
  std::shared_ptr<spmv::DeviceExecutor> exec
      = spmv::SyclExecutor::create(&queue);
  spmv::Matrix<double> A = spmv::read_petsc_binary_matrix(argv1, MPI_COMM_WORLD,
                                                          exec, symmetric, cm);

  auto timer_end = std::chrono::system_clock::now();
  timings["0.MatCreate"] += (timer_end - timer_start);

  // Get local and global sizes
  std::shared_ptr<const spmv::L2GMap> l2g = A.col_map();
  std::int64_t M = A.row_map()->local_size();
  std::int64_t N = l2g->global_size();

  // Vector with extra space for ghosts at end
  if (mpi_rank == 0)
    std::cout << "Creating vector of size " << N << "\n";

  timer_start = std::chrono::system_clock::now();
  double* psp = sycl::malloc_shared<double>(
      l2g->local_size() + l2g->num_ghosts(), queue);

  // Set up values in local range
  int r0 = l2g->global_offset();
  for (int i = 0; i < M; ++i) {
    double z = (double)(i + r0) / double(N);
    psp[i] = exp(-10 * pow(5 * (z - 0.5), 2.0));
  }

  timer_end = std::chrono::system_clock::now();
  timings["1.VecCreate"] += (timer_end - timer_start);

  // Apply matrix a few times
  int n_apply = 100;
  if (mpi_rank == 0)
    std::cout << "Applying matrix " << n_apply << " times\n";

  // Temporary vector
  double* q = sycl::malloc_shared<double>(M, queue);
  double pnorm_local;

  // Warm-up
  l2g->update(psp);
  A.mult(psp, q);
  exec->synchronize();

  for (int i = 0; i < n_apply; ++i) {

    MPI_Barrier(MPI_COMM_WORLD);
    timer_start = std::chrono::system_clock::now();
    l2g->update(psp);
    timer_end = std::chrono::system_clock::now();
    timings["2.SpUpdate"] += (timer_end - timer_start);

    MPI_Barrier(MPI_COMM_WORLD);
    timer_start = std::chrono::system_clock::now();
    A.mult(psp, q);
    exec->synchronize();
    timer_end = std::chrono::system_clock::now();
    timings["3.SpMV"] += (timer_end - timer_start);

    MPI_Barrier(MPI_COMM_WORLD);
    timer_start = std::chrono::system_clock::now();
    queue.copy(q, psp, M).wait();
    timer_end = std::chrono::system_clock::now();
    timings["4.Copy"] += (timer_end - timer_start);
  }

  spmv::squared_norm(M, psp, &pnorm_local, queue);

  double pnorm;
  MPI_Allreduce(&pnorm_local, &pnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  pnorm = sqrt(pnorm);

  if (mpi_rank == 0)
    std::cout << "\nTimings (" << mpi_size
              << ")\n----------------------------\n";

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

  if (mpi_rank == 0) {
    std::cout << "----------------------------\n";
    std::cout << "norm = " << pnorm << "\n";
  }

  sycl::free(psp, queue);
  sycl::free(q, queue);
}

//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED) {
    std::cout << "The threading support level is lesser than required"
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  spmv_main(argc, argv);

  MPI_Finalize();
  return 0;
}

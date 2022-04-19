// Copyright (C) 2018-2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <chrono>
#include <iostream>
#include <memory>

#include <mpi.h>

#include <spmv/spmv.h>
#include <sycl/blas_sycl.h>

using namespace std;

int cg_main(int argc, char** argv)
{
  // Turn off profiling
  MPI_Pcontrol(0);

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Keep list of timings
  map<string, chrono::duration<double>> timings;

  auto timer_start = chrono::system_clock::now();

  string argv1, argv2;
  if (argc == 3) {
    argv1 = argv[1];
    argv2 = argv[2];
  } else {
    throw runtime_error("Use: ./cg_demo <matrix_file> <vector_file>");
  }

  sycl::queue queue(sycl::cpu_selector{});
  auto device = queue.get_device();
  cout << "\n[INFO]: MPI rank " << mpi_rank << " running on "
       << device.get_info<sycl::info::device::name>() << endl;

  // Read matrix
  bool symmetric = false;
  spmv::CommunicationModel cm = spmv::CommunicationModel::collective_blocking;
  std::shared_ptr<spmv::SyclExecutor> exec = spmv::SyclExecutor::create(&queue);
  spmv::Matrix<double> A = spmv::read_petsc_binary_matrix(argv1, MPI_COMM_WORLD,
                                                          exec, symmetric, cm);

  // Read vector
  double* b = spmv::read_petsc_binary_vector(MPI_COMM_WORLD, exec, argv2);
  double* x = exec->alloc<double>(A.rows());

  // Get local and global sizes
  shared_ptr<const spmv::L2GMap> l2g = A.col_map();
  int64_t N = l2g->global_size();

  if (mpi_rank == 0)
    cout << "Global vec size = " << N << "\n";

  auto timer_end = chrono::system_clock::now();
  timings["0.ReadPetsc"] += (timer_end - timer_start);

  int max_its = 100;
  double rtol = 1e-10;

  // Turn on profiling for solver only
  MPI_Pcontrol(1);
  timer_start = chrono::system_clock::now();
  // Solve A*x = b
  int num_its = spmv::cg(MPI_COMM_WORLD, *exec, A, b, x, max_its, rtol);
  timer_end = chrono::system_clock::now();
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
  spmv::axpy<double>(M, -1, b, r, queue).wait();
  double rnorm;
  spmv::dot<double>(M, r, r, &rnorm, queue).wait();
  double rnorm_sum;
  MPI_Allreduce(&rnorm, &rnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Get norm of solution vector
  double xnorm;
  spmv::dot<double>(M, x, x, &xnorm, queue).wait();
  double xnorm_sum;
  MPI_Allreduce(&xnorm, &xnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    cout << "r.norm = " << sqrt(rnorm_sum) << "\n";
    cout << "x.norm = " << sqrt(xnorm_sum) << " in " << num_its
         << " iterations\n";
    cout << "\nTimings (" << mpi_size << ")\n----------------------------\n";
  }

  chrono::duration<double> total_time = chrono::duration<double>::zero();
  for (auto q : timings)
    total_time += q.second;
  timings["Total"] = total_time;

  for (auto q : timings) {
    double q_local = q.second.count(), q_max, q_min;
    MPI_Reduce(&q_local, &q_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&q_local, &q_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
      string pad(16 - q.first.size(), ' ');
      cout << "[" << q.first << "]" << pad << q_min << '\t' << q_max << "\n";
    }
  }

  if (mpi_rank == 0)
    cout << "----------------------------\n";

  // Cleanup
  exec->free(b);
  exec->free(x);
  exec->free(x_padded);
  exec->free(r);

  return 0;
}
//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  cg_main(argc, argv);

  MPI_Finalize();
  return 0;
}

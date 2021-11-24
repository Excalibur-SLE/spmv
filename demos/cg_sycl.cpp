// Copyright (C) 2018-2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>

#include <spmv/spmv.h>

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
  bool use_symmetry = true;
  auto A = spmv::read_petsc_binary_matrix(MPI_COMM_WORLD, argv1, use_symmetry);

  // Read vector
  auto b = spmv::read_petsc_binary_vector(MPI_COMM_WORLD, argv2);

  // Get local and global sizes
  shared_ptr<const spmv::L2GMap> l2g = A.col_map();
  int64_t N = l2g->global_size();
  int N_padded = l2g->local_size() + l2g->num_ghosts();

  if (mpi_rank == 0)
    cout << "Global vec size = " << N << "\n";

  auto timer_end = chrono::system_clock::now();
  timings["0.ReadPetsc"] += (timer_end - timer_start);

  int max_its = 10;
  double rtol = 1e-10;

  // Turn on profiling for solver only
  MPI_Pcontrol(1);
  timer_start = chrono::system_clock::now();
  // Solve A*x = b
  auto [x, num_its]
      = spmv::cg(MPI_COMM_WORLD, queue, A, b.data(), max_its, rtol);
  timer_end = chrono::system_clock::now();
  timings["1.Solve"] += (timer_end - timer_start);
  MPI_Pcontrol(0);

  // Test result
  double xnorm, xnorm_sum;
  double rnorm, rnorm_sum;
  l2g->update(x);
  {
    // Compute norm of x
    sycl::buffer<double> x_buf{x, sycl::range(N_padded)};
    spmv::squared_norm(queue, l2g->local_size(), x_buf, &xnorm);
    MPI_Allreduce(&xnorm, &xnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Compute norm of r = A*x - b
    sycl::buffer<double> y_buf{sycl::range(A.rows())};
    A.mult(queue, x_buf, y_buf);

    // r = A*x - b
    sycl::buffer<double> b_buf{b.data(), sycl::range(A.rows())};
    sycl::buffer<double> r_buf{sycl::range(A.rows())};
    spmv::axpy(queue, A.rows(), -1.0, b_buf, y_buf, r_buf);
    spmv::squared_norm(queue, l2g->local_size(), r_buf, &rnorm);
    MPI_Allreduce(&rnorm, &rnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

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

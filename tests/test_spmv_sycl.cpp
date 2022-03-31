// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <set>

#include <spmv.h>

// Compare double precision floating-point numbers (taken from the "Art of
// Computer Programming")
bool essentially_equal(double a, double b, double epsilon)
{
  return fabs(a - b) <= ((fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

// Divide range into ~equal chunks
static std::vector<int> compute_ranges(int size, int N)
{
  // Compute number of items per process and remainder
  const int n = N / size;
  const int r = N % size;

  // Compute local range
  std::vector<int> ranges;
  for (int rank = 0; rank < (size + 1); ++rank) {
    if (rank < r)
      ranges.push_back(rank * (n + 1));
    else
      ranges.push_back(rank * n + r);
  }

  return ranges;
}

static bool test_spmv(bool symmetric, sycl::queue& queue)
{
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Define a global N x N matrix
  const int N = 5;
  const int NNZ = 15;
  int rowptr[N + 1] = {0, 3, 6, 9, 13, 15};

  int colind[NNZ] = {0, 1, 3, 0, 1, 3, 2, 3, 4, 0, 1, 2, 3, 2, 4};

  double values[NNZ] = {1.0,  -2.0, -3.0, -2.0, 5.0, 4.0,  6.0, 4.0,
                        -4.0, -3.0, 4.0,  4.0,  8.0, -4.0, 8.0};

  // Define a global input vector
  Eigen::VectorXd x(N);
  for (int i = 0; i < N; ++i) {
    double z = (double)(i) / double(N);
    x[i] = exp(-10 * pow(5 * (z - 0.5), 2.0));
  }

  // Compute reference result sequentially
  Eigen::VectorXd y_ref(N);
  for (int i = 0; i < N; ++i) {
    y_ref(i) = 0.0;
    for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      y_ref(i) += values[j] * x(colind[j]);
    }
  }
  double norm_ref = y_ref.norm();

  // Divide matrix by rows evenly across processes
  std::vector<int> ranges = compute_ranges(mpi_size, N);
  int row_start = ranges[mpi_rank];
  int row_end = ranges[mpi_rank + 1];
  int nrows_local = row_end - row_start;
  int ncols_local = nrows_local;
  int nnz_local = rowptr[row_end] - rowptr[row_start];

  // Define local part of the matrix
  int* rowptr_local = rowptr + row_start;
  int* colind_local = colind + rowptr[row_start];
  double* values_local = values + rowptr[row_start];

  // Adapt colind to local indexing
  // Determine column ghosts and remap corresponding indices to re-defined local
  // indices
  std::set<int> ghost_indices;
  for (int i = 0; i < nnz_local; ++i) {
    int global_index = colind_local[i];
    if (global_index < row_start || global_index >= row_end) {
      ghost_indices.insert(global_index);
    }
  }
  std::vector<std::int64_t> col_ghosts(ghost_indices.begin(),
                                       ghost_indices.end());

  // Need mapping of global to local column indices
  for (int i = 0; i < nnz_local; ++i) {
    int global_index = colind_local[i];
    if (global_index < row_start || global_index >= row_end) {
      // Binary search in col_ghosts to find local index
      auto it = std::lower_bound(col_ghosts.begin(), col_ghosts.end(),
                                 global_index);
      colind_local[i] = ncols_local + std::distance(col_ghosts.begin(), it);
    } else {
      colind_local[i] -= row_start;
    }
  }

  // Adapt rowptr to local indexing
  int row_offset = rowptr[row_start];
  for (int i = 0; i < nrows_local + 1; i++)
    rowptr_local[i] -= row_offset;

  // Define SYCL device executor
  std::shared_ptr<spmv::DeviceExecutor> exec
      = spmv::SyclExecutor::create(&queue);
  // Create matrix
  spmv::Matrix<double>* A = spmv::Matrix<double>::create_matrix(
      MPI_COMM_WORLD, exec, rowptr_local, colind_local, values_local,
      nrows_local, ncols_local, {}, col_ghosts, symmetric);

  // Define local vectors
  std::shared_ptr<const spmv::L2GMap> l2g = A->col_map();
  auto y_local = sycl::malloc_shared<double>(nrows_local, queue);
  auto x_local = sycl::malloc_shared<double>(
      l2g->local_size() + l2g->num_ghosts(), queue);
  // Initialize from global vector
  memcpy(x_local, x.data() + row_start, ncols_local * sizeof(double));

  // Update input vector from neighbors
  l2g->update(x_local);

  // Compute SpMV
  A->mult(x_local, y_local);

  double norm_test;
  {
    Eigen::Map<Eigen::VectorXd> y_local_tmp(y_local, nrows_local);
    double norm = y_local_tmp.squaredNorm();
    MPI_Allreduce(&norm, &norm_test, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    norm_test = sqrt(norm_test);
  }

  // Cleanup
  delete A;
  sycl::free(x_local, queue);
  sycl::free(y_local, queue);

  if (essentially_equal(norm_test, norm_ref,
                        std::numeric_limits<double>::epsilon())) {
    std::cout << "PASSED (Rank " << mpi_rank << ")" << std::endl;
    return true;
  } else {
    std::cout << "FAILED (Rank " << mpi_rank << ")" << std::endl;
    return false;
  }
}

int main(int argc, char** argv)
{
#ifdef _OPENMP
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED) {
    std::cerr << "The threading support level is lesser than required"
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
#else
  MPI_Init(&argc, &argv);
#endif

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  sycl::queue cpu_queue(sycl::cpu_selector{});

  bool ret = true;
  bool symmetric;

  symmetric = false;
  if (mpi_rank == 0)
    std::cout << "Running vanilla SpMV on CPU... " << std::endl;
  ret &= test_spmv(symmetric, cpu_queue);
  MPI_Barrier(MPI_COMM_WORLD);

  symmetric = true;
  if (mpi_rank == 0)
    std::cout << "Running symmetric SpMV on CPU... " << std::endl;
  ret &= test_spmv(symmetric, cpu_queue);

  // Run only if a is GPU available
  sycl::queue gpu_queue(sycl::gpu_selector{});
  sycl::device dev = gpu_queue.get_device();
  if (dev.is_gpu()) {
    symmetric = false;
    if (mpi_rank == 0)
      std::cout << "Running vanilla SpMV on GPU... " << std::endl;
    ret &= test_spmv(symmetric, gpu_queue);

    symmetric = true;
    if (mpi_rank == 0)
      std::cout << "Running symmetric SpMV on GPU... " << std::endl;
    ret &= test_spmv(symmetric, gpu_queue);
  }

  MPI_Finalize();
  return (ret) ? 0 : 1;
}

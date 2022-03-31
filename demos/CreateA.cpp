// Copyright (C) 2018-2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <Eigen/Sparse>
#include <memory>
#include <set>

#include "CreateA.h"
#include <spmv/spmv.h>

//-----------------------------------------------------------------------------
// Divide size into N ~equal chunks
std::vector<std::int64_t> owner_ranges(std::int64_t size, std::int64_t N)
{
  // Compute number of items per process and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  // Compute local range
  std::vector<std::int64_t> ranges;
  for (int rank = 0; rank < (size + 1); ++rank) {
    if (rank < r)
      ranges.push_back(rank * (n + 1));
    else
      ranges.push_back(rank * n + r);
  }

  return ranges;
}
//-----------------------------------------------------------------------------
spmv::Matrix<double> create_A(MPI_Comm comm, int N)
{
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(comm, &mpi_size);

  // Make a square Matrix divided evenly across cores
  std::vector<std::int64_t> ranges = owner_ranges(mpi_size, N);

  std::int64_t r0 = ranges[mpi_rank];
  std::int64_t r1 = ranges[mpi_rank + 1];
  int M = r1 - r0;

  // Local part of the matrix
  // Must be RowMajor and compressed
  Eigen::SparseMatrix<double, Eigen::RowMajor> A(M, N);

  // Set up A
  // Add entries on all local rows
  // Using [local_row, global_column] indexing
  double gamma = 0.1;
  for (int i = 0; i < M; ++i) {
    // Global column diagonal index
    int c0 = r0 + i;
    // Special case for very first and last global rows
    if (c0 == 0) {
      A.insert(i, c0) = 1.0 - gamma;
      A.insert(i, c0 + 1) = gamma;
    } else if (c0 == (N - 1)) {
      A.insert(i, c0 - 1) = gamma;
      A.insert(i, c0) = 1.0 - gamma;
    } else {
      A.insert(i, c0 - 1) = gamma;
      A.insert(i, c0) = 1.0 - 2.0 * gamma;
      A.insert(i, c0 + 1) = gamma;
    }
  }
  A.makeCompressed();

  // Remap columns to local indexing
  std::set<std::int64_t> ghost_indices;
  std::int32_t nnz = A.outerIndexPtr()[M];
  for (std::int32_t i = 0; i < nnz; ++i) {
    std::int32_t global_index = A.innerIndexPtr()[i];
    if (global_index < r0 or global_index >= r1)
      ghost_indices.insert(global_index);
  }

  std::vector<std::int64_t> ghosts(ghost_indices.begin(), ghost_indices.end());
  std::shared_ptr<spmv::ReferenceExecutor> exec
      = spmv::ReferenceExecutor::create();
  auto col_l2g = std::make_shared<spmv::L2GMap>(comm, M, ghosts, exec);
  auto row_l2g = std::make_shared<spmv::L2GMap>(
      comm, M, std::vector<std::int64_t>(), exec);

  // Rebuild A using local indices
  Eigen::SparseMatrix<double, Eigen::RowMajor> Alocal(M, M + ghosts.size());
  std::vector<Eigen::Triplet<double>> vals;
  std::int32_t* Aouter = A.outerIndexPtr();
  std::int32_t* Ainner = A.innerIndexPtr();
  double* Aval = A.valuePtr();

  for (std::int32_t row = 0; row < M; ++row) {
    for (std::int32_t j = Aouter[row]; j < Aouter[row + 1]; ++j) {
      std::int32_t col = col_l2g->global_to_local(Ainner[j]);
      double val = Aval[j];
      vals.push_back(Eigen::Triplet<double>(row, col, val));
    }
  }
  Alocal.setFromTriplets(vals.begin(), vals.end());

  return spmv::Matrix<double>(Alocal, col_l2g, row_l2g, exec);
}
//-----------------------------------------------------------------------------

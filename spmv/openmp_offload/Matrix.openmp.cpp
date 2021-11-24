// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

//-----------------------------------------------------------------------------
#pragma omp declare target
void merge_path_search(const int diagonal, const int nrows, const int nnz,
                       const int* rowptr, int* row_idx, int* val_idx)
{
  int row_min = std::max(diagonal - nnz, 0);
  int row_max = std::min(diagonal, nrows);

  // Binary search constraint: row_idx + val_idx = diagonal
  // We are looking for the row_idx for which we can consume "diagonal" number
  // of elements from both the rowptr and values array
  while (row_min < row_max) {
    int pivot = row_min + (row_max - row_min) / 2;
    // The total number of elements I have consumed from both the rowptr and
    // values array at row_idx==pivot is equal to the sum of rowptr[pivot + 1]
    // (number of nonzeros including this row) and (pivot + 1) (the number of
    // entries from rowptr)
    if (rowptr[pivot + 1] + pivot + 1 <= diagonal) {
      // Move downwards and discard top right of cross diagonal range
      row_min = pivot + 1;
    } else {
      // Move upwards and discard bottom left of cross diagonal range
      row_max = pivot;
    }
  }

  *row_idx = row_min;
  *val_idx = diagonal - row_min;
}
#pragma omp end declare target
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::mult(T* x, T* y) const
{
  if (_symmetric) {
    spmv_sym_openmp_offload(x, y);
    return;
  }
  spmv_openmp_offload(x, y);
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::spmv_openmp_offload(const T* x, T* y) const
{
  const int nrows = _mat_local->rows();
  const int ncols = _mat_local->cols();
  const int* rowptr = _mat_local->outerIndexPtr();
  const int* colind = _mat_local->innerIndexPtr();
  const T* values = _mat_local->valuePtr();

  // For GCC compiler, teams+parallel map to warps/wavefronts and simd maps to
  // threads/work items CUDA kernel launched: dim={#teams,1,1},
  // blocks={#threads,warp_size,1}
  // For PGI compiler, teams+parallel map to thread blocks and simd is not used
  // Clang/LLVM does not implement simd
  #pragma omp target teams distribute parallel for schedule(static, 1)	\
    map(to: x[:ncols]) \
    map(from: y[:nrows])
  for (int i = 0; i < nrows; ++i) {
    if (i < nrows) {
      T y_tmp = 0.0;

      for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        y_tmp += values[j] * x[colind[j]];
      }

      y[i] = y_tmp;
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::spmv_sym_openmp_offload(const T* x, T* y) const
{
  const int nrows = _mat_local->rows();
  const int ncols = _mat_local->cols();
  const int* rowptr = _mat_local->outerIndexPtr();
  const int* colind = _mat_local->innerIndexPtr();
  const T* values = _mat_local->valuePtr();
  const int nnz_rmt = _mat_remote->nonZeros();
  const int* rowptr_rmt = _mat_remote->outerIndexPtr();
  const int* colind_rmt = _mat_remote->innerIndexPtr();
  const T* values_rmt = _mat_remote->valuePtr();
  const T* diagonal = _mat_diagonal->data();

  // Naive implementation based on local vectors (one per team)
  // Distribute blocks of rows to teams
  #pragma omp target teams distribute dist_schedule(static, BLOCK_SIZE)	\
    map(to: nrows, _nblocks)						\
    map(to: x[:ncols])							\
    map(from: y[:nrows])						\
    num_teams(_nblocks)
  for (int b = 0; b < _nblocks; ++b) {

    // Distribute rows in a block to threads
    #pragma omp parallel for num_threads(TEAM_SIZE)
    for (int i = b * BLOCK_SIZE; i < min((b + 1) * BLOCK_SIZE, nrows); ++i) {
      T y_tmp = diagonal[i] * x[i];

      // Compute symmetric SpMV on local block
      for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        int col = colind[j];
        T val = values[j];
        y_tmp += val * x[col];
        T tmp = val * x[i];
        #pragma omp atomic
        y[col] += tmp;
      }

      // Compute vanilla SpMV on remote block
      if (nnz_rmt > 0) {
        for (int j = rowptr_rmt[i]; j < rowptr_rmt[i + 1]; ++j) {
          y_tmp += values_rmt[j] * x[colind_rmt[j]];
        }
      }

      y[i] += y_tmp;
    }
  }
}
//-----------------------------------------------------------------------------

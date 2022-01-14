// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

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

  memset(y, 0, nrows * sizeof(T));

  // For GCC compiler, teams+parallel map to warps/wavefronts and simd maps to
  // threads/work items CUDA kernel launched: dim={#teams,1,1},
  // blocks={#threads,warp_size,1}
  // For PGI compiler, teams+parallel map to thread blocks and simd is not used
  // Clang/LLVM does not implement simd
  #pragma omp target teams distribute parallel for schedule(static, 1)	\
    map(to: x[:ncols])							\
    map(tofrom: y[:nrows])
  for (int i = 0; i < nrows; ++i) {
    T y_tmp = diagonal[i] * x[i];

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
    #pragma omp atomic
    y[i] += y_tmp;
  }
}
//-----------------------------------------------------------------------------

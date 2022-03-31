// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "csr_kernels.h"
#include "omp_executor.h"
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <unordered_set>

#ifdef _MKL
#include <mkl.h>
#endif // _MKL

namespace spmv
{

// #ifdef _MKL
//   sparse_matrix_t _mat_local_mkl;
//   sparse_matrix_t _mat_remote_mkl;
//   struct matrix_descr _mat_local_desc;
//   struct matrix_descr _mat_remote_desc;
// #endif // _MKL

struct aux_data_t {
  int32_t* _row_split = nullptr;
  int* _cnfl_pos = nullptr;
  short* _cnfl_src = nullptr;
  int* _cnfl_start = nullptr;
  int* _cnfl_end = nullptr;
  int _ncnfls = 0;
  char* _buffer = nullptr;
};

template <typename T>
void CSRSpMV<T>::init(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
                      int32_t* rowptr, int32_t* colind, T* values,
                      bool symmetric, const OmpExecutor& exec)
{
  _symmetric = symmetric;

  if (num_non_zeros > 0) {
    // FIXME
    //  _aux_data = exec.alloc<aux_data_t>(1);
    _aux_data = new aux_data_t;
    aux_data_t* aux_data = (aux_data_t*)_aux_data;
    int num_threads = exec.get_num_cus();

    // partition_by_nnz(nthreads);
    aux_data->_row_split = exec.alloc<int32_t>(num_threads + 1);
    int32_t* row_split = aux_data->_row_split;
    if (num_threads == 1) {
      row_split[0] = 0;
      row_split[1] = num_rows;
      return;
    }

    // Compute the matrix splits.
    int nnz_per_split = (num_non_zeros + num_threads - 1) / num_threads;
    int curr_nnz = 0;
    int row_start = 0;
    int split_cnt = 0;
    int i;

    row_split[0] = row_start;
    for (i = 0; i < num_rows; i++) {
      curr_nnz += rowptr[i + 1] - rowptr[i];
      if (curr_nnz >= nnz_per_split) {
        row_start = i + 1;
        ++split_cnt;
        if (split_cnt <= num_threads)
          row_split[split_cnt] = row_start;
        curr_nnz = 0;
      }
    }

    // Fill the last split with remaining elements
    if (curr_nnz < nnz_per_split && split_cnt <= num_threads) {
      row_split[++split_cnt] = num_rows;
    }

    // If there are any remaining rows merge them in last partition
    if (split_cnt > num_threads) {
      row_split[num_threads] = num_rows;
    }

    // If there are remaining threads create empty partitions
    for (int i = split_cnt + 1; i <= num_threads; i++) {
      row_split[i] = num_rows;
    }

    // Compute conflict map
    if (_symmetric) {
      // Allocate buffers for "local vectors indexing" method
      // The first thread writes directly to the output vector, so doesn't need
      // a buffer
      aux_data->_buffer = (char*)exec.alloc<T>(num_threads * num_rows);

      // Build conflict map for local block
      std::map<int, std::unordered_set<int>> row_conflicts;
      std::set<int> thread_conflicts;
      int ncnfls = 0;
      for (int tid = 1; tid < num_threads; ++tid) {
        for (int i = row_split[tid]; i < row_split[tid + 1]; ++i) {
          for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            int target_row = colind[j];
            if (target_row < row_split[tid]) {
              thread_conflicts.insert(target_row);
              row_conflicts[target_row].insert(tid);
            }
          }
        }
        ncnfls += thread_conflicts.size();
        thread_conflicts.clear();
      }

      // Finalise conflict map data structure
      aux_data->_cnfl_pos = exec.alloc<int>(ncnfls);
      aux_data->_cnfl_src = exec.alloc<short>(ncnfls);
      int32_t* cnfl_pos = aux_data->_cnfl_pos;
      short* cnfl_src = aux_data->_cnfl_src;
      int cnt = 0;
      for (auto& conflict : row_conflicts) {
        for (auto tid : conflict.second) {
          cnfl_pos[cnt] = conflict.first;
          cnfl_src[cnt] = tid;
          cnt++;
        }
      }
      assert(cnt == ncnfls);

      // Split reduction work among threads so that conflicts to the same row
      // are assigned to the same thread
      aux_data->_cnfl_start = exec.alloc<int>(num_threads);
      aux_data->_cnfl_end = exec.alloc<int>(num_threads);
      int32_t* cnfl_start = aux_data->_cnfl_start;
      int32_t* cnfl_end = aux_data->_cnfl_end;
      int total_count = ncnfls;
      int tid = 0;
      int limit = total_count / num_threads;
      int tmp_count = 0, run_cnt = 0;
      for (auto& elem : row_conflicts) {
        run_cnt += elem.second.size();
        if (tmp_count < limit) {
          tmp_count += elem.second.size();
        } else {
          cnfl_end[tid] = tmp_count;
          // If we have exceeded the number of threads, assigned what is left to
          // last thread
          total_count -= tmp_count;
          tmp_count = elem.second.size();
          limit = total_count / (num_threads - (tid + 1));
          tid++;
          if (tid == num_threads - 1) {
            break;
          }
        }
      }

      for (int i = tid; i < num_threads; i++)
        cnfl_end[i] = ncnfls - (run_cnt - tmp_count);

      int start = 0;
      for (int tid = 0; tid < num_threads; tid++) {
        cnfl_start[tid] = start;
        cnfl_end[tid] += start;
        start = cnfl_end[tid];
      }

      aux_data->_ncnfls = ncnfls;
    }
  }
}

template <typename T>
void CSRSpMV<T>::run(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
                     const int32_t* rowptr, const int32_t* colind,
                     const T* values, const T* diagonal, T alpha,
                     T* __restrict__ in, T beta, T* __restrict__ out,
                     const OmpExecutor& exec) const
{
  if (_symmetric && num_non_zeros > 0) {
    aux_data_t* aux_data = static_cast<aux_data_t*>(_aux_data);
    int32_t* row_split = aux_data->_row_split;
    short* cnfl_src = aux_data->_cnfl_src;
    int32_t* cnfl_pos = aux_data->_cnfl_pos;
    int32_t* cnfl_start = aux_data->_cnfl_start;
    int32_t* cnfl_end = aux_data->_cnfl_end;
    T* buffer = (T*)aux_data->_buffer;

// Assumes out is initialized to zero if beta is zero
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      const int32_t row_offset = row_split[tid];
      T* buffer = (T*)aux_data->_buffer;

      // Local vectors phase
      for (int i = row_split[tid]; i < row_split[tid + 1]; ++i) {
        T sum = diagonal[i] * in[i];

        if (rowptr != nullptr) {
          for (int32_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            int32_t col = colind[j];
            T val = values[j];
            sum += val * in[col];
            if (col < row_offset) {
              buffer[tid * num_rows + col] += val * in[i];
            } else {
              out[col] += alpha * val * in[i];
            }
          }
        }
        out[i] = alpha * sum + beta * out[i];
      }
#pragma omp barrier

      // Reduction of conflicts phase
      for (int i = cnfl_start[tid]; i < cnfl_end[tid]; ++i) {
        int vid = cnfl_src[i];
        int pos = cnfl_pos[i];
        out[pos] += alpha * buffer[vid * num_rows + pos];
        buffer[vid * num_rows + pos] = 0.0;
      }
    }
  } else if (_symmetric) {
#pragma omp parallel for
    for (int32_t i = 0; i < num_rows; i++)
      out[i] = alpha * diagonal[i] * in[i] + beta * out[i];
  } else {
#pragma omp parallel
    {
      aux_data_t* aux_data = static_cast<aux_data_t*>(_aux_data);
      int32_t* row_split = aux_data->_row_split;
      const int tid = omp_get_thread_num();
      const int row_offset = row_split[tid];

      for (int32_t i = row_split[tid]; i < row_split[tid + 1]; ++i) {
        T sum = 0.0;

        for (int32_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
          sum += values[j] * in[colind[j]];
        }

        out[i] = alpha * sum + beta * out[i];
      }
    }
  }
}

template <typename T>
void CSRSpMV<T>::finalize(const OmpExecutor& exec) const
{
  aux_data_t* aux_data = (aux_data_t*)_aux_data;
  if (aux_data != nullptr) {
    exec.free(aux_data->_row_split);
    if (_symmetric && aux_data->_ncnfls > 0) {
      exec.free(aux_data->_cnfl_pos);
      exec.free(aux_data->_cnfl_src);
      exec.free(aux_data->_cnfl_start);
      exec.free(aux_data->_cnfl_end);
      exec.free(aux_data->_buffer);
    }
    //  exec.free(aux_data);
    delete aux_data;
  }
}

} // namespace spmv

// Explicit instantiations
template class spmv::CSRSpMV<float>;
template class spmv::CSRSpMV<double>;

// template <typename T>
// void Matrix<T>::partition_by_nrows(const int nthreads)
// {
//   if (!_row_split) {
//     _row_split = new int[nthreads + 1];
//   }

//   int nrows = _mat_local->rows();
//   if (nthreads == 1) {
//     _row_split[0] = 0;
//     _row_split[1] = nrows;
//     return;
//   }

//   // Compute new matrix splits
//   int nrows_per_split = (nrows + nthreads - 1) / nthreads;
//   int i;
//   _row_split[0] = 0;
//   for (i = 0; i < nthreads; i++) {
//     if (_row_split[i] + nrows_per_split < nrows) {
//       _row_split[i + 1] = _row_split[i] + nrows_per_split;
//     } else {
//       _row_split[i + 1] = _row_split[i] + nrows - i * nrows_per_split;
//       break;
//     }
//   }

//   for (int j = i; j <= nthreads; j++) {
//     _row_split[j] = nrows;
//   }
// }
// //-----------------------------------------------------------------------------
// template <typename T>
// void Matrix<T>::partition_by_nnz(const int nthreads)
// {
//   // const int nrows = _mat_local->rows();
//   // const int nnz = _mat_local->nonZeros() + _mat_remote->nonZeros();
//   // const int* rowptr = _mat_local->outerIndexPtr();
//   // const int* rowptr_outer = _mat_remote->outerIndexPtr();

//   // if (!_row_split) {
//   //   _row_split = new int[nthreads + 1];
//   // }

//   // if (nthreads == 1) {
//   //   _row_split[0] = 0;
//   //   _row_split[1] = nrows;
//   //   return;
//   // }

//   // // Compute the matrix splits.
//   // int nnz_per_split = (nnz + nthreads - 1) / nthreads;
//   // int curr_nnz = 0;
//   // int row_start = 0;
//   // int split_cnt = 0;
//   // int i;

//   // _row_split[0] = row_start;
//   // for (i = 0; i < nrows; i++) {
//   //   curr_nnz
//   //       += rowptr[i + 1] - rowptr[i] + rowptr_outer[i + 1] -
//   rowptr_outer[i];
//   //   if (curr_nnz >= nnz_per_split) {
//   //     row_start = i + 1;
//   //     ++split_cnt;
//   //     if (split_cnt <= nthreads)
//   //       _row_split[split_cnt] = row_start;
//   //     curr_nnz = 0;
//   //   }
//   // }

//   // // Fill the last split with remaining elements
//   // if (curr_nnz < nnz_per_split && split_cnt <= nthreads) {
//   //   _row_split[++split_cnt] = nrows;
//   // }

//   // // If there are any remaining rows merge them in last partition
//   // if (split_cnt > nthreads) {
//   //   _row_split[nthreads] = nrows;
//   // }

//   // // If there are remaining threads create empty partitions
//   // for (int i = split_cnt + 1; i <= nthreads; i++) {
//   //   _row_split[i] = nrows;
//   // }
// }

// #ifdef _MKL
// template <>
// void Matrix<double>::mkl_init()
// {
//   sparse_status_t status = mkl_sparse_d_create_csr(
//       &_mat_local_mkl, SPARSE_INDEX_BASE_ZERO, _mat_local->rows(),
//       _mat_local->cols(), const_cast<MKL_INT*>(_mat_local->outerIndexPtr()),
//       const_cast<MKL_INT*>(_mat_local->outerIndexPtr()) + 1,
//       const_cast<MKL_INT*>(_mat_local->innerIndexPtr()),
//       const_cast<double*>(_mat_local->valuePtr()));
//   assert(status == SPARSE_STATUS_SUCCESS);
//   if (status != SPARSE_STATUS_SUCCESS)
//     throw runtime_error("Could not create MKL matrix");

//   status = mkl_sparse_optimize(_mat_local_mkl);
//   assert(status == SPARSE_STATUS_SUCCESS);
//   if (status != SPARSE_STATUS_SUCCESS)
//     throw runtime_error("Could not optimize MKL matrix");

//   _mat_local_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
//   _mat_local_desc.diag = SPARSE_DIAG_NON_UNIT;

//   if (_col_map->overlapping() && _mat_remote->nonZeros() > 0) {
//     status = mkl_sparse_d_create_csr(
//         &_mat_remote_mkl, SPARSE_INDEX_BASE_ZERO, _mat_remote->rows(),
//         _mat_remote->cols(),
//         const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()),
//         const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()) + 1,
//         const_cast<MKL_INT*>(_mat_remote->innerIndexPtr()),
//         const_cast<double*>(_mat_remote->valuePtr()));
//     assert(status == SPARSE_STATUS_SUCCESS);
//     if (status != SPARSE_STATUS_SUCCESS)
//       throw runtime_error("Could not create MKL matrix");

//     status = mkl_sparse_optimize(_mat_remote_mkl);
//     assert(status == SPARSE_STATUS_SUCCESS);
//     if (status != SPARSE_STATUS_SUCCESS)
//       throw runtime_error("Could not optimize MKL matrix");

//     _mat_remote_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
//     _mat_remote_desc.diag = SPARSE_DIAG_NON_UNIT;
//   }
// }
// //-----------------------------------------------------------------------------
// template <>
// void Matrix<complex<double>>::mkl_init()
// {
//   sparse_status_t status = mkl_sparse_z_create_csr(
//       &_mat_local_mkl, SPARSE_INDEX_BASE_ZERO, _mat_local->rows(),
//       _mat_local->cols(), const_cast<MKL_INT*>(_mat_local->outerIndexPtr()),
//       const_cast<MKL_INT*>(_mat_local->outerIndexPtr()) + 1,
//       const_cast<MKL_INT*>(_mat_local->innerIndexPtr()),
//       (MKL_Complex16*)_mat_local->valuePtr());
//   assert(status == SPARSE_STATUS_SUCCESS);
//   if (status != SPARSE_STATUS_SUCCESS)
//     throw runtime_error("Could not create MKL matrix");

//   status = mkl_sparse_optimize(_mat_local_mkl);
//   assert(status == SPARSE_STATUS_SUCCESS);
//   if (status != SPARSE_STATUS_SUCCESS)
//     throw runtime_error("Could not optimize MKL matrix");

//   _mat_local_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
//   _mat_local_desc.diag = SPARSE_DIAG_NON_UNIT;

//   if (_col_map->overlapping() && _mat_remote->nonZeros() > 0) {
//     status = mkl_sparse_z_create_csr(
//         &_mat_remote_mkl, SPARSE_INDEX_BASE_ZERO, _mat_remote->rows(),
//         _mat_remote->cols(),
//         const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()),
//         const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()) + 1,
//         const_cast<MKL_INT*>(_mat_remote->innerIndexPtr()),
//         (MKL_Complex16*)_mat_remote->valuePtr());
//     assert(status == SPARSE_STATUS_SUCCESS);
//     if (status != SPARSE_STATUS_SUCCESS)
//       throw runtime_error("Could not create MKL matrix");

//     status = mkl_sparse_optimize(_mat_remote_mkl);
//     assert(status == SPARSE_STATUS_SUCCESS);
//     if (status != SPARSE_STATUS_SUCCESS)
//       throw runtime_error("Could not optimize MKL matrix");

//     _mat_remote_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
//     _mat_remote_desc.diag = SPARSE_DIAG_NON_UNIT;
//   }
// }
// //-----------------------------------------------------------------------------
// template <>
// void Matrix<float>::mkl_init()
// {
//   sparse_status_t status = mkl_sparse_s_create_csr(
//       &_mat_local_mkl, SPARSE_INDEX_BASE_ZERO, _mat_local->rows(),
//       _mat_local->cols(), const_cast<MKL_INT*>(_mat_local->outerIndexPtr()),
//       const_cast<MKL_INT*>(_mat_local->outerIndexPtr()) + 1,
//       const_cast<MKL_INT*>(_mat_local->innerIndexPtr()),
//       const_cast<float*>(_mat_local->valuePtr()));
//   assert(status == SPARSE_STATUS_SUCCESS);
//   if (status != SPARSE_STATUS_SUCCESS)
//     throw runtime_error("Could not create MKL matrix");

//   status = mkl_sparse_optimize(_mat_local_mkl);
//   assert(status == SPARSE_STATUS_SUCCESS);
//   if (status != SPARSE_STATUS_SUCCESS)
//     throw runtime_error("Could not optimize MKL matrix");

//   _mat_local_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
//   _mat_local_desc.diag = SPARSE_DIAG_NON_UNIT;

//   if (_col_map->overlapping() && _mat_remote->nonZeros() > 0) {
//     status = mkl_sparse_s_create_csr(
//         &_mat_remote_mkl, SPARSE_INDEX_BASE_ZERO, _mat_remote->rows(),
//         _mat_remote->cols(),
//         const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()),
//         const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()) + 1,
//         const_cast<MKL_INT*>(_mat_remote->innerIndexPtr()),
//         const_cast<float*>(_mat_remote->valuePtr()));
//     assert(status == SPARSE_STATUS_SUCCESS);
//     if (status != SPARSE_STATUS_SUCCESS)
//       throw runtime_error("Could not create MKL matrix");

//     status = mkl_sparse_optimize(_mat_remote_mkl);
//     assert(status == SPARSE_STATUS_SUCCESS);
//     if (status != SPARSE_STATUS_SUCCESS)
//       throw runtime_error("Could not optimize MKL matrix");

//     _mat_remote_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
//     _mat_remote_desc.diag = SPARSE_DIAG_NON_UNIT;
//   }
// }
// //-----------------------------------------------------------------------------
// template <>
// void Matrix<complex<float>>::mkl_init()
// {
//   sparse_status_t status = mkl_sparse_c_create_csr(
//       &_mat_local_mkl, SPARSE_INDEX_BASE_ZERO, _mat_local->rows(),
//       _mat_local->cols(), const_cast<MKL_INT*>(_mat_local->outerIndexPtr()),
//       const_cast<MKL_INT*>(_mat_local->outerIndexPtr()) + 1,
//       const_cast<MKL_INT*>(_mat_local->innerIndexPtr()),
//       (MKL_Complex8*)_mat_local->valuePtr());
//   assert(status == SPARSE_STATUS_SUCCESS);
//   if (status != SPARSE_STATUS_SUCCESS)
//     throw runtime_error("Could not create MKL matrix");

//   status = mkl_sparse_optimize(_mat_local_mkl);
//   assert(status == SPARSE_STATUS_SUCCESS);
//   if (status != SPARSE_STATUS_SUCCESS)
//     throw runtime_error("Could not optimize MKL matrix");

//   _mat_local_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
//   _mat_local_desc.diag = SPARSE_DIAG_NON_UNIT;

//   if (_col_map->overlapping() && _mat_remote->nonZeros() > 0) {
//     status = mkl_sparse_c_create_csr(
//         &_mat_remote_mkl, SPARSE_INDEX_BASE_ZERO, _mat_remote->rows(),
//         _mat_remote->cols(),
//         const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()),
//         const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()) + 1,
//         const_cast<MKL_INT*>(_mat_remote->innerIndexPtr()),
//         (MKL_Complex8*)_mat_remote->valuePtr());
//     assert(status == SPARSE_STATUS_SUCCESS);
//     if (status != SPARSE_STATUS_SUCCESS)
//       throw runtime_error("Could not create MKL matrix");

//     status = mkl_sparse_optimize(_mat_remote_mkl);
//     assert(status == SPARSE_STATUS_SUCCESS);
//     if (status != SPARSE_STATUS_SUCCESS)
//       throw runtime_error("Could not optimize MKL matrix");

//     _mat_remote_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
//     _mat_remote_desc.diag = SPARSE_DIAG_NON_UNIT;
//   }
// }
// //-----------------------------------------------------------------------------
// template <>
// Eigen::VectorXd Matrix<double>::mult(Eigen::VectorXd& b) const
// {
//   Eigen::VectorXd y(_mat_local->rows());
//   if (_symmetric) {
//     y = spmv_sym(b);
//   } else if (_col_map->overlapping() && _mat_remote->nonZeros() > 0) {
//     mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mat_local_mkl,
//                     _mat_local_desc, b.data(), 0.0, y.data());
//     _col_map->update_finalise(b.data());
//     mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mat_remote_mkl,
//                     _mat_remote_desc, b.data(), 1.0, y.data());
//   } else {
//     mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mat_local_mkl,
//                     _mat_local_desc, b.data(), 0.0, y.data());
//   }

//   return y;
// }
// //-----------------------------------------------------------------------------
// template <>
// Eigen::VectorXd Matrix<double>::transpmult(const Eigen::VectorXd& b) const
// {
//   Eigen::VectorXd y(_mat_local->cols());
//   if (_symmetric) {
//     throw runtime_error(
//         "transpmult() operation not yet implemented for symmetric matrices");
//   } else {
//     mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, _mat_local_mkl,
//                     _mat_local_desc, b.data(), 0.0, y.data());
//   }

//   return y;
// }
// //-----------------------------------------------------------------------------
// template <>
// Eigen::Matrix<complex<double>, Eigen::Dynamic, 1>
// Matrix<complex<double>>::mult(
//     Eigen::Matrix<complex<double>, Eigen::Dynamic, 1>& b) const
// {
//   Eigen::Matrix<complex<double>, Eigen::Dynamic, 1> y(_mat_local->rows());
//   const MKL_Complex16 one({1.0, 0.0}), zero({0.0, 0.0});
//   if (_symmetric) {
//     throw runtime_error("Multiplication not yet implemented for symmetric "
//                         "matrices for complex data");
//   } else if (_col_map->overlapping() && _mat_remote->nonZeros() > 0) {
//     mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, one, _mat_local_mkl,
//                     _mat_local_desc, (MKL_Complex16*)b.data(), zero,
//                     (MKL_Complex16*)y.data());
//     _col_map->update_finalise(b.data());
//     mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, one, _mat_remote_mkl,
//                     _mat_remote_desc, (MKL_Complex16*)b.data(), one,
//                     (MKL_Complex16*)y.data());
//   } else {
//     mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, one, _mat_local_mkl,
//                     _mat_local_desc, (MKL_Complex16*)b.data(), zero,
//                     (MKL_Complex16*)y.data());
//   }

//   return y;
// }
// //-----------------------------------------------------------------------------
// template <>
// Eigen::Matrix<complex<double>, Eigen::Dynamic, 1>
// Matrix<complex<double>>::transpmult(
//     const Eigen::Matrix<complex<double>, Eigen::Dynamic, 1>& b) const
// {
//   Eigen::Matrix<complex<double>, Eigen::Dynamic, 1> y(_mat_local->rows());
//   const MKL_Complex16 one({1.0, 0.0}), zero({0.0, 0.0});
//   if (_symmetric) {
//     throw runtime_error(
//         "transpmult() operation not yet implemented for symmetric matrices");
//   } else {
//     mkl_sparse_z_mv(SPARSE_OPERATION_TRANSPOSE, one, _mat_local_mkl,
//                     _mat_local_desc, (MKL_Complex16*)b.data(), zero,
//                     (MKL_Complex16*)y.data());
//   }
//   return y;
// }
// //-----------------------------------------------------------------------------
// template <>
// Eigen::VectorXf Matrix<float>::mult(Eigen::VectorXf& b) const
// {
//   Eigen::VectorXf y(_mat_local->rows());
//   if (_symmetric) {
//     y = spmv_sym(b);
//   } else if (_col_map->overlapping() && _mat_remote->nonZeros() > 0) {
//     mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mat_local_mkl,
//                     _mat_local_desc, b.data(), 0.0, y.data());
//     _col_map->update_finalise(b.data());
//     mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mat_remote_mkl,
//                     _mat_remote_desc, b.data(), 1.0, y.data());
//   } else {
//     mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mat_local_mkl,
//                     _mat_local_desc, b.data(), 0.0, y.data());
//   }

//   return y;
// }
// //-----------------------------------------------------------------------------
// template <>
// Eigen::VectorXf Matrix<float>::transpmult(const Eigen::VectorXf& b) const
// {
//   Eigen::VectorXf y(_mat_local->cols());
//   if (_symmetric) {
//     throw runtime_error(
//         "transpmult() operation not yet implemented for symmetric matrices");
//   } else {
//     mkl_sparse_s_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, _mat_local_mkl,
//                     _mat_local_desc, b.data(), 0.0, y.data());
//   }

//   return y;
// }
// //-----------------------------------------------------------------------------
// template <>
// Eigen::Matrix<complex<float>, Eigen::Dynamic, 1>
// Matrix<complex<float>>::mult(
//     Eigen::Matrix<complex<float>, Eigen::Dynamic, 1>& b) const
// {
//   Eigen::Matrix<complex<float>, Eigen::Dynamic, 1> y(_mat_local->rows());
//   const MKL_Complex8 one({1.0, 0.0}), zero({0.0, 0.0});
//   if (_symmetric) {
//     throw runtime_error("Multiplication not yet implemented for symmetric "
//                         "matrices for complex data");
//   } else if (_col_map->overlapping() && _mat_remote->nonZeros() > 0) {
//     mkl_sparse_c_mv(SPARSE_OPERATION_NON_TRANSPOSE, one, _mat_local_mkl,
//                     _mat_local_desc, (MKL_Complex8*)b.data(), zero,
//                     (MKL_Complex8*)y.data());
//     _col_map->update_finalise(b.data());
//     mkl_sparse_c_mv(SPARSE_OPERATION_NON_TRANSPOSE, one, _mat_remote_mkl,
//                     _mat_remote_desc, (MKL_Complex8*)b.data(), one,
//                     (MKL_Complex8*)y.data());
//   } else {
//     mkl_sparse_c_mv(SPARSE_OPERATION_NON_TRANSPOSE, one, _mat_local_mkl,
//                     _mat_local_desc, (MKL_Complex8*)b.data(), zero,
//                     (MKL_Complex8*)y.data());
//   }

//   return y;
// }
// //-----------------------------------------------------------------------------
// template <>
// Eigen::Matrix<complex<float>, Eigen::Dynamic, 1>
// Matrix<complex<float>>::transpmult(
//     const Eigen::Matrix<complex<float>, Eigen::Dynamic, 1>& b) const
// {
//   Eigen::Matrix<complex<float>, Eigen::Dynamic, 1> y(_mat_local->rows());
//   const MKL_Complex8 one({1.0, 0.0}), zero({0.0, 0.0});
//   if (_symmetric) {
//     throw runtime_error(
//         "transpmult() operation not yet implemented for symmetric matrices");
//   } else {
//     mkl_sparse_c_mv(SPARSE_OPERATION_TRANSPOSE, one, _mat_local_mkl,
//                     _mat_local_desc, (MKL_Complex8*)b.data(), zero,
//                     (MKL_Complex8*)y.data());
//   }
//   return y;
// }
// #endif // _MKL

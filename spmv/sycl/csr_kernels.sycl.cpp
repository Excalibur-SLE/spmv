// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "csr_kernels.h"
#include "sycl_executor.h"
#include <cassert>
#include <map>
#include <set>
#include <unordered_set>

namespace spmv
{

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
                      bool symmetric, const SyclExecutor& exec)
{
  _symmetric = symmetric;
  _aux_data = new aux_data_t;
  aux_data_t* aux_data = (aux_data_t*)_aux_data;
  sycl::queue* q = exec.get_queue();
  auto device = q->get_device();

  if (num_non_zeros > 0) {
    if (device.is_cpu()) {
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
        // The first thread writes directly to the output vector, so doesn't
        // need a buffer
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
            // If we have exceeded the number of threads, assigned what is left
            // to last thread
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
}

template <typename T>
void CSRSpMV<T>::run(int32_t num_rows, int32_t num_cols, int32_t num_non_zeros,
                     const int32_t* rowptr, const int32_t* colind,
                     const T* values, const T* diagonal, T alpha,
                     T* __restrict__ in, T beta, T* __restrict__ out,
                     const SyclExecutor& exec) const
{
  sycl::queue* q = exec.get_queue();
  auto device = q->get_device();
  if (device.is_cpu()) {
    int num_threads = exec.get_num_cus();
    aux_data_t* aux_data = (aux_data_t*)_aux_data;
    int32_t* row_split = aux_data->_row_split;

    if (_symmetric && num_non_zeros > 0) {
      short* cnfl_src = aux_data->_cnfl_src;
      int32_t* cnfl_pos = aux_data->_cnfl_pos;
      int32_t* cnfl_start = aux_data->_cnfl_start;
      int32_t* cnfl_end = aux_data->_cnfl_end;
      T* buffer = (T*)aux_data->_buffer;

      // Local vectors phase
      q->submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range(num_threads), [=](sycl::id<1> it) {
           const int32_t tid = it[0];
           const int32_t row_offset = row_split[tid];
           for (int32_t i = row_split[tid]; i < row_split[tid + 1]; ++i) {
             T sum = diagonal[i] * in[i];

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

             out[i] = alpha * sum + beta * out[i];
           }
         });
       }).wait();

      // Reduction of local vectors phase
      if (aux_data->_ncnfls > 0) {
        q->submit([&](sycl::handler& h) {
           h.parallel_for(sycl::range(num_threads), [=](sycl::id<1> it) {
             const int32_t tid = it[0];
             for (int32_t i = cnfl_start[tid]; i < cnfl_end[tid]; ++i) {
               int32_t vid = cnfl_src[i];
               int32_t pos = cnfl_pos[i];
               out[pos] += alpha * buffer[vid * num_rows + pos];
               buffer[vid * num_rows + pos] = 0.0;
             }
           });
         }).wait();
      }
    } else if (_symmetric) {
      q->submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range(num_rows), [=](sycl::id<1> it) {
           const int32_t tid = it[0];
           out[tid] = alpha * diagonal[tid] * in[tid] + beta * out[tid];
         });
       }).wait();
    } else {
      q->submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range(num_threads), [=](sycl::id<1> it) {
           const int32_t tid = it[0];
           const int32_t row_offset = row_split[tid];
           for (int32_t i = row_split[tid]; i < row_split[tid + 1]; ++i) {
             T sum = 0;

             for (int32_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
               sum += values[j] * in[colind[j]];
             }

             out[i] = alpha * sum + beta * out[i];
           }
         });
       }).wait();
    }
  } else if (device.is_gpu()) {
    if (_symmetric && num_non_zeros > 0) {
      // Scale by beta
      q->submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range(num_rows), [=](sycl::id<1> it) {
           const int32_t tid = it[0];
           out[tid] = beta * out[tid];
         });
       }).wait();

      q->submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range(num_rows), [=](sycl::id<1> it) {
           const int32_t i = it[0];
           T sum = diagonal[i] * in[i];

           for (int32_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
             int32_t col = colind[j];
             T val = values[j];
             sum += val * in[col];
#ifdef __HIPSYCL__
             sycl::atomic_ref<T, sycl::memory_order::relaxed,
                              sycl::memory_scope::system>
                 atomic_out(out[col]);
#else
             sycl::ext::oneapi::atomic_ref<T, sycl::memory_order::relaxed,
                                      sycl::memory_scope::system,
                                      sycl::access::address_space::global_space>
                 atomic_out(out[col]);
#endif
             atomic_out += alpha * val * in[i];
           }

#ifdef __HIPSYCL__
           sycl::atomic_ref<T, sycl::memory_order::relaxed,
                            sycl::memory_scope::system>
               atomic_out(out[i]);
#else
           sycl::ext::oneapi::atomic_ref<T, sycl::memory_order::relaxed,
                                    sycl::memory_scope::system,
                                    sycl::access::address_space::global_space>
               atomic_out(out[i]);
#endif
           atomic_out += alpha * sum;
         });
       }).wait();
    } else if (_symmetric) {
      q->submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range(num_rows), [=](sycl::id<1> it) {
           const int32_t tid = it[0];
           out[tid] = alpha * diagonal[tid] * in[tid] + beta * out[tid];
         });
       }).wait();
    } else {
      q->submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range(num_rows), [=](sycl::id<1> it) {
           const int32_t i = it[0];
           T sum = 0;

           for (int32_t j = rowptr[i]; j < rowptr[i + 1]; ++j) {
             sum += values[j] * in[colind[j]];
           }

           out[i] = alpha * sum + beta * out[i];
         });
       }).wait();
    }
  }
}

template <typename T>
void CSRSpMV<T>::finalize(const SyclExecutor& exec) const
{
  aux_data_t* aux_data = (aux_data_t*)_aux_data;
  sycl::queue* q = exec.get_queue();
  auto device = q->get_device();

  if (device.is_cpu()) {
    if (aux_data != nullptr) {
      exec.free(aux_data->_row_split);
      if (_symmetric) {
        exec.free(aux_data->_cnfl_pos);
        exec.free(aux_data->_cnfl_src);
        exec.free(aux_data->_cnfl_start);
        exec.free(aux_data->_cnfl_end);
        exec.free(aux_data->_buffer);
      }
      delete aux_data;
    }
  }
}

} // namespace spmv

// Explicit instantiations
template class spmv::CSRSpMV<float>;
template class spmv::CSRSpMV<double>;

// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

//-----------------------------------------------------------------------------
static MergeCoordinate merge_path_search(const int diagonal, const int nrows,
                                         const int nnz, const int* rowptr)
{
  int row_min = sycl::max(diagonal - nnz, 0);
  int row_max = sycl::min(diagonal, nrows);

  // Binary search constraint: row_idx + val_idx = diagonal
  // We are looking for the row_idx for which we can consume "diagonal" number
  // of elements from both the rowptr and values array
  while (row_min < row_max) {
    int pivot = row_min + (row_max - row_min) / 2;
    // The total number of elements I have consumed from both the rowptr and
    // values array at row_idx==pivot is equal to the sum of rowptr[pivot + 1]
    // (number of nonzeros including this row) and (pivot + 1) (the number of
    // entries from rowptr)
    if (pivot < nrows) {
      if (rowptr[pivot + 1] + pivot + 1 <= diagonal) {
        // Move downwards and discard top right of cross diagonal range
        row_min = pivot + 1;
      } else {
        // Move upwards and discard bottom left of cross diagonal range
        row_max = pivot;
      }
    }
  }

  MergeCoordinate path_coordinate;
  path_coordinate.row_idx = sycl::min(row_min, nrows);
  path_coordinate.val_idx = diagonal - row_min;
  return path_coordinate;
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::tune(sycl::queue& queue)
{
  namespace acc = sycl::access;
  auto device = queue.get_device();

  if (device.is_gpu() && !_symmetric) {
    // Precompute merge coordinates
    const int nrows = _mat_local->rows();
    const int nnz = _mat_local->nonZeros();
    int merge_path_length = nrows + nnz;
    int num_work_groups
        = device.get_info<sycl::info::device::max_compute_units>();
    int work_group_size
        = device.get_info<sycl::info::device::max_work_group_size>();
    int num_work_items = num_work_groups * work_group_size;
    int items_per_work_item
        = (merge_path_length + num_work_items - 1) / num_work_items;
    _merge_path = new MergeCoordinate[num_work_items + 1];
    _carry_row = new int[num_work_groups];
    _carry_val = new T[num_work_groups];

    // Initialise SYCL buffers, ownership is passed to SYCL runtime
    // Use the provided host pointer and do not allocate new data on the host
    auto properties
        = sycl::property_list{sycl::property::buffer::use_host_ptr()};
    _d_merge_path = new sycl::buffer(
        _merge_path, sycl::range(num_work_items + 1), properties);
    _d_carry_row = new sycl::buffer(_carry_row, sycl::range(num_work_groups),
                                    properties);
    _d_carry_val = new sycl::buffer(_carry_val, sycl::range(num_work_groups),
                                    properties);
    {
      queue.submit([&](sycl::handler& cgh) {
        auto rowptr
            = _d_rowptr_local->template get_access<acc::mode::read>(cgh);
        auto merge_path
            = _d_merge_path->template get_access<acc::mode::write>(cgh);

        cgh.parallel_for(
            sycl::nd_range(sycl::range(num_work_items),
                           sycl::range(work_group_size)),
            [=](sycl::nd_item<1> item) {
              int tid = item.get_global_id(0);
              if (tid < num_work_items) {
                // Find starting merge path coordinates (row index
                // and value index) for this thread int diagonal_start =
                // min(items_per_thread * tid, merge_path_length); int
                // diagonal_end = min(items_per_thread * (tid + 1),
                // merge_path_length);
                int diagonal_start
                    = ((items_per_work_item * tid) < merge_path_length)
                          ? (items_per_work_item * tid)
                          : merge_path_length;
                MergeCoordinate path = merge_path_search(
                    diagonal_start, nrows, nnz, rowptr.get_pointer());
                merge_path[tid].val_idx = path.val_idx;
                merge_path[tid].row_idx = path.row_idx;
              }
            });
      });
      queue.wait();
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::mult(sycl::buffer<T>& x_buf, sycl::buffer<T>& y_buf,
                     sycl::queue& queue) const
{
  if (_symmetric)
    spmv_sym_sycl(x_buf, y_buf, queue);
  else
    spmv_sycl(x_buf, y_buf, queue);
}
//-----------------------------------------------------------------------------
template <typename T>
sycl::event Matrix<T>::mult(T* x, T* y, sycl::queue& queue,
                            const std::vector<sycl::event>& dependencies) const
{
  if (_symmetric)
    return spmv_sym_sycl(x, y, queue, dependencies);
  else
    return spmv_sycl(x, y, queue, dependencies);
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::spmv_sycl(sycl::buffer<T>& x_buf, sycl::buffer<T>& y_buf,
                          sycl::queue& queue) const
{
  namespace acc = sycl::access;
  auto device = queue.get_device();

  if (device.is_gpu()) {
    // The workgroup size must divide the ND-range size exactly in each
    // dimension
    const int nrows = _mat_local->rows();
    const int nnz = _mat_local->nonZeros();
    int merge_path_length = nrows + nnz;
    int num_work_groups
        = device.get_info<sycl::info::device::max_compute_units>();
    int work_group_size
        = device.get_info<sycl::info::device::max_work_group_size>();
    int num_work_items = num_work_groups * work_group_size;
    int items_per_work_item
        = (merge_path_length + num_work_items - 1) / num_work_items;

    using local_accessor_index_t
        = sycl::accessor<int, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>;
    using local_accessor_value_t
        = sycl::accessor<T, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>;
    {
      queue.submit([&](sycl::handler& h) {
        sycl::accessor rowptr{*_d_rowptr_local, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_local, h, sycl::read_only};
        sycl::accessor values{*_d_values_local, h, sycl::read_only};
        sycl::accessor merge_path{*_d_merge_path, h, sycl::read_only};
        sycl::accessor global_carry_row{*_d_carry_row, h, sycl::read_write};
        sycl::accessor global_carry_val{*_d_carry_val, h, sycl::read_write};
        sycl::accessor x{x_buf, h, sycl::read_only};
#ifdef __HIPSYCL__
        sycl::accessor y{y_buf, h, sycl::write_only, sycl::no_init};
#else
        sycl::accessor y{y_buf, h, sycl::write_only, sycl::noinit};
#endif
        auto local_carry_row = local_accessor_index_t(work_group_size, h);
        auto local_carry_val = local_accessor_value_t(work_group_size, h);

        h.parallel_for(
            sycl::nd_range<1>(num_work_items, work_group_size),
            [=](sycl::nd_item<1> item) {
              int global_id = item.get_global_id(0);
              if (global_id < num_work_items) {
                int local_id = item.get_local_id(0);
                int work_group_id = item.get_group(0);
                int work_group_size = item.get_local_range(0);

                MergeCoordinate path_start = merge_path[global_id];
                MergeCoordinate path_end = merge_path[global_id + 1];

                T sum = 0;
                for (int i = 0; i < items_per_work_item; i++) {
                  if (path_start.row_idx < nrows) {
                    if (path_start.val_idx < rowptr[path_start.row_idx + 1]) {
                      // Accumulate and move down
                      sum += values[path_start.val_idx]
                             * x[colind[path_start.val_idx]];
                      path_start.val_idx++;
                    } else {
                      // Flush row and move right
                      if (path_start.row_idx < nrows) {
                        y[path_start.row_idx] = sum;
                        sum = 0;
                        path_start.row_idx++;
                      }
                    }
                  }
                }

                // Save carry
                local_carry_row[local_id] = path_end.row_idx;
                local_carry_val[local_id] = sum;

                // Wait for all threads to put their data in local memory
                item.barrier(sycl::access::fence_space::local_space);

                // Reduce local carry values
                // This is a segmented scan pattern, should find good
                // implementation
                if (local_id == 0) {
                  for (int i = 0; i < work_group_size - 1; i++) {
                    if (local_carry_row[i]
                        != local_carry_row[work_group_size - 1]) {
                      y[local_carry_row[i]] += local_carry_val[i];
                    } else {
                      local_carry_val[work_group_size - 1]
                          += local_carry_val[i];
                    }
                  }
                }

                // Wait for all threads to put their data in local memory
                item.barrier(sycl::access::fence_space::local_space);

                // Last threads stores global carry
                if (local_id == work_group_size - 1) {
                  global_carry_row[work_group_id] = local_carry_row[local_id];
                  global_carry_val[work_group_id] = local_carry_val[local_id];
                }
              }
            });
      });

      queue.submit([&](sycl::handler& h) {
        sycl::accessor global_carry_row{*_d_carry_row, h, sycl::read_only};
        sycl::accessor global_carry_val{*_d_carry_val, h, sycl::read_only};
        sycl::accessor y{y_buf, h, sycl::read_write};
        h.parallel_for(sycl::range<1>{static_cast<size_t>(num_work_groups)},
                       [=](sycl::id<1> it) {
                         const int wg = it[0];

                         if (global_carry_row[wg] < nrows) {
#ifdef __HIPSYCL__
                           sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                            sycl::memory_scope::system>
                               atomic_y(y[global_carry_row[wg]]);
#else
                           sycl::ONEAPI::atomic_ref<
                               T, sycl::ONEAPI::memory_order::relaxed,
                               sycl::ONEAPI::memory_scope::system,
                               sycl::access::address_space::global_space>
                               atomic_y(y[global_carry_row[wg]]);
#endif
                           atomic_y += global_carry_val[wg];
                         }
                       });
      });
    }
  } else {
    queue.submit([&](sycl::handler& h) {
      const size_t nrows = _mat_local->rows();
      sycl::accessor rowptr{*_d_rowptr_local, h, sycl::read_only};
      sycl::accessor colind{*_d_colind_local, h, sycl::read_only};
      sycl::accessor values{*_d_values_local, h, sycl::read_only};
      sycl::accessor x{x_buf, h, sycl::read_only};
#ifdef __HIPSYCL__
      sycl::accessor y{y_buf, h, sycl::write_only, sycl::no_init};
#else
      sycl::accessor y{y_buf, h, sycl::write_only, sycl::noinit};
#endif

      h.parallel_for(sycl::range(nrows), [=](sycl::id<1> it) {
        const int i = it[0];
        T sum = 0;

        for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
          sum += values[j] * x[colind[j]];
        }

        y[i] = sum;
      });
    });
  }
}
//-----------------------------------------------------------------------------
template <typename T>
sycl::event
Matrix<T>::spmv_sycl(T* x, T* y, sycl::queue& queue,
                     const std::vector<sycl::event>& dependencies) const
{
  namespace acc = sycl::access;
  auto device = queue.get_device();
  sycl::event e, prev_e;

  if (device.is_gpu()) {
    // The workgroup size must divide the ND-range size exactly in each
    // dimension
    const int nrows = _mat_local->rows();
    const int nnz = _mat_local->nonZeros();
    int merge_path_length = nrows + nnz;
    int num_work_groups
        = device.get_info<sycl::info::device::max_compute_units>();
    int work_group_size
        = device.get_info<sycl::info::device::max_work_group_size>();
    int num_work_items = num_work_groups * work_group_size;
    int items_per_work_item
        = (merge_path_length + num_work_items - 1) / num_work_items;

    using local_accessor_index_t
        = sycl::accessor<int, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>;
    using local_accessor_value_t
        = sycl::accessor<T, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>;
    {
      e = queue.submit([&](sycl::handler& h) {
        h.depends_on(dependencies);
        sycl::accessor rowptr{*_d_rowptr_local, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_local, h, sycl::read_only};
        sycl::accessor values{*_d_values_local, h, sycl::read_only};
        sycl::accessor merge_path{*_d_merge_path, h, sycl::read_only};
        sycl::accessor global_carry_row{*_d_carry_row, h, sycl::read_write};
        sycl::accessor global_carry_val{*_d_carry_val, h, sycl::read_write};
        auto local_carry_row = local_accessor_index_t(work_group_size, h);
        auto local_carry_val = local_accessor_value_t(work_group_size, h);

        h.parallel_for(
            sycl::nd_range<1>(num_work_items, work_group_size),
            [=](sycl::nd_item<1> item) {
              int global_id = item.get_global_id(0);
              if (global_id < num_work_items) {
                int local_id = item.get_local_id(0);
                int work_group_id = item.get_group(0);
                int work_group_size = item.get_local_range(0);

                MergeCoordinate path_start = merge_path[global_id];
                MergeCoordinate path_end = merge_path[global_id + 1];

                T sum = 0;
                for (int i = 0; i < items_per_work_item; i++) {
                  if (path_start.row_idx < nrows) {
                    if (path_start.val_idx < rowptr[path_start.row_idx + 1]) {
                      // Accumulate and move down
                      sum += values[path_start.val_idx]
                             * x[colind[path_start.val_idx]];
                      path_start.val_idx++;
                    } else {
                      // Flush row and move right
                      if (path_start.row_idx < nrows) {
                        y[path_start.row_idx] = sum;
                        sum = 0;
                        path_start.row_idx++;
                      }
                    }
                  }
                }

                // Save carry
                local_carry_row[local_id] = path_end.row_idx;
                local_carry_val[local_id] = sum;

                // Wait for all threads to put their data in local memory
                item.barrier(sycl::access::fence_space::local_space);

                // Reduce local carry values
                // This is a segmented scan pattern, should find good
                // implementation
                if (local_id == 0) {
                  for (int i = 0; i < work_group_size - 1; i++) {
                    if (local_carry_row[i]
                        != local_carry_row[work_group_size - 1]) {
                      y[local_carry_row[i]] += local_carry_val[i];
                    } else {
                      local_carry_val[work_group_size - 1]
                          += local_carry_val[i];
                    }
                  }
                }

                // Wait for all threads to put their data in local memory
                item.barrier(sycl::access::fence_space::local_space);

                // Last threads stores global carry
                if (local_id == work_group_size - 1) {
                  global_carry_row[work_group_id] = local_carry_row[local_id];
                  global_carry_val[work_group_id] = local_carry_val[local_id];
                }
              }
            });
      });
      prev_e = e;

      e = queue.submit([&](sycl::handler& h) {
        h.depends_on({prev_e});
        sycl::accessor global_carry_row{*_d_carry_row, h, sycl::read_only};
        sycl::accessor global_carry_val{*_d_carry_val, h, sycl::read_only};
        h.parallel_for(sycl::range<1>{static_cast<size_t>(num_work_groups)},
                       [=](sycl::id<1> it) {
                         const int wg = it[0];

                         if (global_carry_row[wg] < nrows) {
#ifdef __HIPSYCL__
                           sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                            sycl::memory_scope::system>
                               atomic_y(y[global_carry_row[wg]]);
#else
                           sycl::ONEAPI::atomic_ref<
                               T, sycl::ONEAPI::memory_order::relaxed,
                               sycl::ONEAPI::memory_scope::system,
                               sycl::access::address_space::global_space>
                               atomic_y(y[global_carry_row[wg]]);
#endif
                           atomic_y += global_carry_val[wg];
                         }
                       });
      });
    }
  } else {
    e = queue.submit([&](sycl::handler& h) {
      h.depends_on(dependencies);
      const size_t nrows = _mat_local->rows();
      sycl::accessor rowptr{*_d_rowptr_local, h, sycl::read_only};
      sycl::accessor colind{*_d_colind_local, h, sycl::read_only};
      sycl::accessor values{*_d_values_local, h, sycl::read_only};

      h.parallel_for(sycl::range(nrows), [=](sycl::id<1> it) {
        const int i = it[0];
        T sum = 0;

        for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
          sum += values[j] * x[colind[j]];
        }

        y[i] = sum;
      });
    });
  }

  return e;
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::spmv_sym_sycl(sycl::buffer<T>& x_buf, sycl::buffer<T>& y_buf,
                              sycl::queue& queue) const
{
  namespace acc = sycl::access;
  auto device = queue.get_device();

  if (device.is_gpu()) {
    // Compute diagonal contribution
    queue.submit([&](sycl::handler& h) {
      const size_t nrows = _mat_local->rows();
      sycl::accessor diagonal{*_d_diagonal, h, sycl::read_only};
      sycl::accessor x{x_buf, h, sycl::read_only};
#ifdef __HIPSYCL__
      sycl::accessor y{y_buf, h, sycl::write_only, sycl::no_init};
#else
      sycl::accessor y{y_buf, h, sycl::write_only, sycl::noinit};
#endif

      h.parallel_for(sycl::range(nrows), [=](sycl::id<1> it) {
        const int i = it[0];
        y[i] = diagonal[i] * x[i];
      });
    });

    if (_mat_local->nonZeros() > 0 && _mat_remote->nonZeros() > 0) {
      // Compute symmetric SpMV on local block and vanilla SpMV on remote block
      queue.submit([&](sycl::handler& h) {
        const size_t nrows = _mat_local->rows();
        sycl::accessor rowptr{*_d_rowptr_local, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_local, h, sycl::read_only};
        sycl::accessor values{*_d_values_local, h, sycl::read_only};
        sycl::accessor rowptr_rmt{*_d_rowptr_remote, h, sycl::read_only};
        sycl::accessor colind_rmt{*_d_colind_remote, h, sycl::read_only};
        sycl::accessor values_rmt{*_d_values_remote, h, sycl::read_only};
        sycl::accessor x{x_buf, h, sycl::read_only};
        sycl::accessor y{y_buf, h, sycl::read_write};

        h.parallel_for(sycl::range(nrows), [=](sycl::id<1> it) {
          const int i = it[0];
          T sum = 0;

          for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            int col = colind[j];
            T val = values[j];
            sum += val * x[col];
#ifdef __HIPSYCL__
            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                             sycl::memory_scope::system>
                atomic_y(y[col]);
#else
            sycl::ONEAPI::atomic_ref<T, sycl::ONEAPI::memory_order::relaxed,
                                     sycl::ONEAPI::memory_scope::system,
                                     sycl::access::address_space::global_space>
	        atomic_y(y[col]);
#endif
            atomic_y += val * x[i];
          }

          for (int j = rowptr_rmt[i]; j < rowptr_rmt[i + 1]; ++j) {
            sum += values_rmt[j] * x[colind_rmt[j]];
          }

#ifdef __HIPSYCL__
          sycl::atomic_ref<T, sycl::memory_order::relaxed,
                           sycl::memory_scope::system>
              atomic_y(y[i]);
#else
          sycl::ONEAPI::atomic_ref<T, sycl::ONEAPI::memory_order::relaxed,
                                   sycl::ONEAPI::memory_scope::system,
                                   sycl::access::address_space::global_space>
              atomic_y(y[i]);
#endif
          atomic_y += sum;
        });
      });
    } else if (_mat_local->nonZeros() > 0) {
      // Compute symmetric SpMV on local block
      queue.submit([&](sycl::handler& h) {
        const size_t nrows = _mat_local->rows();
        sycl::accessor rowptr{*_d_rowptr_local, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_local, h, sycl::read_only};
        sycl::accessor values{*_d_values_local, h, sycl::read_only};
        sycl::accessor x{x_buf, h, sycl::read_only};
        sycl::accessor y{y_buf, h, sycl::read_write};

        h.parallel_for(sycl::range(nrows), [=](sycl::id<1> it) {
          const int i = it[0];
          T sum = 0;

          for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            int col = colind[j];
            T val = values[j];
            sum += val * x[col];
#ifdef __HIPSYCL__
            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                             sycl::memory_scope::system>
                atomic_y(y[col]);
#else
            sycl::ONEAPI::atomic_ref<T, sycl::ONEAPI::memory_order::relaxed,
                                     sycl::ONEAPI::memory_scope::system,
                                     sycl::access::address_space::global_space>
                atomic_y(y[col]);
#endif
            atomic_y += val * x[i];
          }

#ifdef __HIPSYCL__
          sycl::atomic_ref<T, sycl::memory_order::relaxed,
                           sycl::memory_scope::system>
              atomic_y(y[i]);
#else
          sycl::ONEAPI::atomic_ref<T, sycl::ONEAPI::memory_order::relaxed,
                                   sycl::ONEAPI::memory_scope::system,
                                   sycl::access::address_space::global_space>
              atomic_y(y[i]);
#endif
          atomic_y += sum;
        });
      });
    } else {
      // Compute vanilla SpMV on remote block
      queue.submit([&](sycl::handler& h) {
        const size_t nrows = _mat_remote->rows();
        sycl::accessor rowptr{*_d_rowptr_remote, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_remote, h, sycl::read_only};
        sycl::accessor values{*_d_values_remote, h, sycl::read_only};
        sycl::accessor x{x_buf, h, sycl::read_only};
        sycl::accessor y{y_buf, h, sycl::read_write};

        h.parallel_for(sycl::range(nrows), [=](sycl::id<1> it) {
          const int i = it[0];
          T sum = 0;

          for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            sum += values[j] * x[colind[j]];
          }

          y[i] += sum;
        });
      });
    }
  } else { // device.is_cpu()
    // Compute diagonal contribution
    queue.submit([&](sycl::handler& h) {
      const size_t nrows = _mat_local->rows();
      sycl::accessor diagonal{*_d_diagonal, h, sycl::read_only};
      sycl::accessor x{x_buf, h, sycl::read_only};
#ifdef __HIPSYCL__
      sycl::accessor y{y_buf, h, sycl::write_only, sycl::no_init};
#else
      sycl::accessor y{y_buf, h, sycl::write_only, sycl::noinit};
#endif

      h.parallel_for(sycl::range(nrows), [=](sycl::id<1> it) {
        const int i = it[0];
        y[i] = diagonal[i] * x[i];
      });
    });

    if (_mat_local->nonZeros() > 0 && _mat_remote->nonZeros() > 0) {
      // Compute symmetric SpMV on local block (local vectors phase) and vanilla
      // SpMV on remote block
      queue.submit([&](sycl::handler& h) {
        sycl::accessor row_split{*_d_row_split, h, sycl::read_only};
        sycl::accessor rowptr{*_d_rowptr_local, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_local, h, sycl::read_only};
        sycl::accessor values{*_d_values_local, h, sycl::read_only};
        sycl::accessor rowptr_rmt{*_d_rowptr_remote, h, sycl::read_only};
        sycl::accessor colind_rmt{*_d_colind_remote, h, sycl::read_only};
        sycl::accessor values_rmt{*_d_values_remote, h, sycl::read_only};
        sycl::accessor x{x_buf, h, sycl::read_only};
        sycl::accessor y{y_buf, h, sycl::read_write};
        sycl::accessor y_local{*_d_y_local, h, sycl::read_write};

        h.parallel_for(sycl::range(_nthreads), [=](sycl::id<1> it) {
          const int tid = it[0];
          const int row_offset = row_split[tid];
          for (int i = row_split[tid]; i < row_split[tid + 1]; ++i) {
            T y_tmp = 0;

            for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
              int col = colind[j];
              T val = values[j];
              y_tmp += val * x[col];
              if (col < row_offset) {
                y_local[tid][col] += val * x[i];
              } else {
                y[col] += val * x[i];
              }
            }

            for (int j = rowptr_rmt[i]; j < rowptr_rmt[i + 1]; ++j) {
              y_tmp += values_rmt[j] * x[colind_rmt[j]];
            }

            y[i] += y_tmp;
          }
        });
      });
    } else if (_mat_local->nonZeros() > 0) {
      // Compute symmetric SpMV on local block
      queue.submit([&](sycl::handler& h) {
        sycl::accessor row_split{*_d_row_split, h, sycl::read_only};
        sycl::accessor rowptr{*_d_rowptr_local, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_local, h, sycl::read_only};
        sycl::accessor values{*_d_values_local, h, sycl::read_only};
        sycl::accessor x{x_buf, h, sycl::read_only};
        sycl::accessor y{y_buf, h, sycl::read_write};
        sycl::accessor y_local{*_d_y_local, h, sycl::read_write};

        h.parallel_for(sycl::range(_nthreads), [=](sycl::id<1> it) {
          const int tid = it[0];
          const int row_offset = row_split[tid];
          for (int i = row_split[tid]; i < row_split[tid + 1]; ++i) {
            T y_tmp = 0;

            for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
              int col = colind[j];
              T val = values[j];
              y_tmp += val * x[col];
              if (col < row_offset) {
                y_local[tid][col] += val * x[i];
              } else {
                y[col] += val * x[i];
              }
            }

            y[i] += y_tmp;
          }
        });
      });
    } else {
      // Compute vanilla SpMV on remote block
      queue.submit([&](sycl::handler& h) {
        sycl::accessor row_split{*_d_row_split, h, sycl::read_only};
        sycl::accessor rowptr{*_d_rowptr_remote, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_remote, h, sycl::read_only};
        sycl::accessor values{*_d_values_remote, h, sycl::read_only};
        sycl::accessor x{x_buf, h, sycl::read_only};
        sycl::accessor y{y_buf, h, sycl::read_write};

        h.parallel_for(sycl::range(_nthreads), [=](sycl::id<1> it) {
          const int tid = it[0];
          for (int i = row_split[tid]; i < row_split[tid + 1]; ++i) {
            T y_tmp = 0;

            for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
              y_tmp += values[j] * x[colind[j]];
            }

            y[i] += y_tmp;
          }
        });
      });
    }

    // Reduction of local vectors phase
    if (_ncnfls > 0) {
      queue.submit([&](sycl::handler& h) {
        sycl::accessor map_start{*_d_map_start, h, sycl::read_only};
        sycl::accessor map_end{*_d_map_end, h, sycl::read_only};
        sycl::accessor cnfl_vid{*_d_cnfl_vid, h, sycl::read_only};
        sycl::accessor cnfl_pos{*_d_cnfl_pos, h, sycl::read_only};
        sycl::accessor y{y_buf, h, sycl::read_write};
        sycl::accessor y_local{*_d_y_local, h, sycl::read_write};

        h.parallel_for(sycl::range(_nthreads), [=](sycl::id<1> it) {
          const int tid = it[0];
          for (int i = map_start[tid]; i < map_end[tid]; ++i) {
            int vid = cnfl_vid[i];
            int pos = cnfl_pos[i];
            y[pos] += y_local[vid][pos];
            y_local[vid][pos] = 0.0;
          }
        });
      });
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
sycl::event
Matrix<T>::spmv_sym_sycl(T* x, T* y, sycl::queue& queue,
                         const std::vector<sycl::event>& dependencies) const
{
  namespace acc = sycl::access;
  auto device = queue.get_device();
  sycl::event e, prev_e;

  if (device.is_gpu()) {
    // Compute diagonal contribution
    e = queue.submit([&](sycl::handler& h) {
      h.depends_on(dependencies);
      const size_t nrows = _mat_local->rows();
      sycl::accessor diagonal{*_d_diagonal, h, sycl::read_only};

      h.parallel_for(sycl::range(nrows), [=](sycl::id<1> it) {
        const int i = it[0];
        y[i] = diagonal[i] * x[i];
      });
    });
    prev_e = e;

    if (_mat_local->nonZeros() > 0 && _mat_remote->nonZeros() > 0) {
      // Compute symmetric SpMV on local block and vanilla SpMV on remote block
      e = queue.submit([&](sycl::handler& h) {
        h.depends_on({prev_e});
        const size_t nrows = _mat_local->rows();
        sycl::accessor rowptr{*_d_rowptr_local, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_local, h, sycl::read_only};
        sycl::accessor values{*_d_values_local, h, sycl::read_only};
        sycl::accessor rowptr_rmt{*_d_rowptr_remote, h, sycl::read_only};
        sycl::accessor colind_rmt{*_d_colind_remote, h, sycl::read_only};
        sycl::accessor values_rmt{*_d_values_remote, h, sycl::read_only};

        h.parallel_for(sycl::range(nrows), [=](sycl::id<1> it) {
          const int i = it[0];
          T sum = 0;

          for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            int col = colind[j];
            T val = values[j];
            sum += val * x[col];
#ifdef __HIPSYCL__
            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                             sycl::memory_scope::system>
                atomic_y(y[col]);
#else
            sycl::ONEAPI::atomic_ref<T, sycl::ONEAPI::memory_order::relaxed,
                                     sycl::ONEAPI::memory_scope::system,
                                     sycl::access::address_space::global_space>
	        atomic_y(y[col]);
#endif
            atomic_y += val * x[i];
          }

          for (int j = rowptr_rmt[i]; j < rowptr_rmt[i + 1]; ++j) {
            sum += values_rmt[j] * x[colind_rmt[j]];
          }

#ifdef __HIPSYCL__
          sycl::atomic_ref<T, sycl::memory_order::relaxed,
                           sycl::memory_scope::system>
              atomic_y(y[i]);
#else
          sycl::ONEAPI::atomic_ref<T, sycl::ONEAPI::memory_order::relaxed,
                                   sycl::ONEAPI::memory_scope::system,
                                   sycl::access::address_space::global_space>
              atomic_y(y[i]);
#endif
          atomic_y += sum;
        });
      });
    } else if (_mat_local->nonZeros() > 0) {
      // Compute symmetric SpMV on local block
      e = queue.submit([&](sycl::handler& h) {
        h.depends_on({prev_e});
        const size_t nrows = _mat_local->rows();
        sycl::accessor rowptr{*_d_rowptr_local, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_local, h, sycl::read_only};
        sycl::accessor values{*_d_values_local, h, sycl::read_only};

        h.parallel_for(sycl::range(nrows), [=](sycl::id<1> it) {
          const int i = it[0];
          T sum = 0;

          for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            int col = colind[j];
            T val = values[j];
            sum += val * x[col];
#ifdef __HIPSYCL__
            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                             sycl::memory_scope::system>
                atomic_y(y[col]);
#else
            sycl::ONEAPI::atomic_ref<T, sycl::ONEAPI::memory_order::relaxed,
                                     sycl::ONEAPI::memory_scope::system,
                                     sycl::access::address_space::global_space>
                atomic_y(y[col]);
#endif
            atomic_y += val * x[i];
          }

#ifdef __HIPSYCL__
          sycl::atomic_ref<T, sycl::memory_order::relaxed,
                           sycl::memory_scope::system>
              atomic_y(y[i]);
#else	  
          sycl::ONEAPI::atomic_ref<T, sycl::ONEAPI::memory_order::relaxed,
                                   sycl::ONEAPI::memory_scope::system,
                                   sycl::access::address_space::global_space>
              atomic_y(y[i]);
#endif
          atomic_y += sum;
        });
      });
    } else {
      // Compute vanilla SpMV on remote block
      e = queue.submit([&](sycl::handler& h) {
        h.depends_on({prev_e});
        const size_t nrows = _mat_remote->rows();
        sycl::accessor rowptr{*_d_rowptr_remote, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_remote, h, sycl::read_only};
        sycl::accessor values{*_d_values_remote, h, sycl::read_only};

        h.parallel_for(sycl::range(nrows), [=](sycl::id<1> it) {
          const int i = it[0];
          T sum = 0;

          for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
            sum += values[j] * x[colind[j]];
          }

          y[i] += sum;
        });
      });
    }
  } else { // device.is_cpu()
    // Compute diagonal contribution
    e = queue.submit([&](sycl::handler& h) {
      h.depends_on(dependencies);
      const size_t nrows = _mat_local->rows();
      sycl::accessor diagonal{*_d_diagonal, h, sycl::read_only};

      h.parallel_for(sycl::range(nrows), [=](sycl::id<1> it) {
        const int i = it[0];
        y[i] = diagonal[i] * x[i];
      });
    });
    prev_e = e;

    if (_mat_local->nonZeros() > 0 && _mat_remote->nonZeros() > 0) {
      // Compute symmetric SpMV on local block (local vectors phase) and vanilla
      // SpMV on remote block
      e = queue.submit([&](sycl::handler& h) {
        h.depends_on({prev_e});
        sycl::accessor row_split{*_d_row_split, h, sycl::read_only};
        sycl::accessor rowptr{*_d_rowptr_local, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_local, h, sycl::read_only};
        sycl::accessor values{*_d_values_local, h, sycl::read_only};
        sycl::accessor rowptr_rmt{*_d_rowptr_remote, h, sycl::read_only};
        sycl::accessor colind_rmt{*_d_colind_remote, h, sycl::read_only};
        sycl::accessor values_rmt{*_d_values_remote, h, sycl::read_only};
        sycl::accessor y_local{*_d_y_local, h, sycl::read_write};

        h.parallel_for(sycl::range(_nthreads), [=](sycl::id<1> it) {
          const int tid = it[0];
          const int row_offset = row_split[tid];
          for (int i = row_split[tid]; i < row_split[tid + 1]; ++i) {
            T y_tmp = 0;

            for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
              int col = colind[j];
              T val = values[j];
              y_tmp += val * x[col];
              if (col < row_offset) {
                y_local[tid][col] += val * x[i];
              } else {
                y[col] += val * x[i];
              }
            }

            for (int j = rowptr_rmt[i]; j < rowptr_rmt[i + 1]; ++j) {
              y_tmp += values_rmt[j] * x[colind_rmt[j]];
            }

            y[i] += y_tmp;
          }
        });
      });
      prev_e = e;
    } else if (_mat_local->nonZeros() > 0) {
      // Compute symmetric SpMV on local block
      e = queue.submit([&](sycl::handler& h) {
        h.depends_on({prev_e});
        sycl::accessor row_split{*_d_row_split, h, sycl::read_only};
        sycl::accessor rowptr{*_d_rowptr_local, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_local, h, sycl::read_only};
        sycl::accessor values{*_d_values_local, h, sycl::read_only};
        sycl::accessor y_local{*_d_y_local, h, sycl::read_write};

        h.parallel_for(sycl::range(_nthreads), [=](sycl::id<1> it) {
          const int tid = it[0];
          const int row_offset = row_split[tid];
          for (int i = row_split[tid]; i < row_split[tid + 1]; ++i) {
            T y_tmp = 0;

            for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
              int col = colind[j];
              T val = values[j];
              y_tmp += val * x[col];
              if (col < row_offset) {
                y_local[tid][col] += val * x[i];
              } else {
                y[col] += val * x[i];
              }
            }

            y[i] += y_tmp;
          }
        });
      });
      prev_e = e;
    } else {
      // Compute vanilla SpMV on remote block
      e = queue.submit([&](sycl::handler& h) {
        h.depends_on({prev_e});
        sycl::accessor row_split{*_d_row_split, h, sycl::read_only};
        sycl::accessor rowptr{*_d_rowptr_remote, h, sycl::read_only};
        sycl::accessor colind{*_d_colind_remote, h, sycl::read_only};
        sycl::accessor values{*_d_values_remote, h, sycl::read_only};

        h.parallel_for(sycl::range(_nthreads), [=](sycl::id<1> it) {
          const int tid = it[0];
          for (int i = row_split[tid]; i < row_split[tid + 1]; ++i) {
            T y_tmp = 0;

            for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
              y_tmp += values[j] * x[colind[j]];
            }

            y[i] += y_tmp;
          }
        });
      });
      prev_e = e;
    }

    // Reduction of local vectors phase
    if (_ncnfls > 0) {
      e = queue.submit([&](sycl::handler& h) {
        h.depends_on({prev_e});
        sycl::accessor map_start{*_d_map_start, h, sycl::read_only};
        sycl::accessor map_end{*_d_map_end, h, sycl::read_only};
        sycl::accessor cnfl_vid{*_d_cnfl_vid, h, sycl::read_only};
        sycl::accessor cnfl_pos{*_d_cnfl_pos, h, sycl::read_only};
        sycl::accessor y_local{*_d_y_local, h, sycl::read_write};

        h.parallel_for(sycl::range(_nthreads), [=](sycl::id<1> it) {
          const int tid = it[0];
          for (int i = map_start[tid]; i < map_end[tid]; ++i) {
            int vid = cnfl_vid[i];
            int pos = cnfl_pos[i];
            y[pos] += y_local[vid][pos];
            y_local[vid][pos] = 0.0;
          }
        });
      });
    }
  }

  return e;
}
//-----------------------------------------------------------------------------

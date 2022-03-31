// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <numeric>
#include <set>
#include <unordered_set>

#include "L2GMap.h"
#include "Matrix.h"
#include "csr_matrix.h"

using namespace spmv;

//-----------------------------------------------------------------------------
template <typename T>
Matrix<T>::Matrix(const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat,
                  shared_ptr<spmv::L2GMap> col_map,
                  shared_ptr<spmv::L2GMap> row_map,
                  std::shared_ptr<DeviceExecutor> exec)
    : _exec(exec), _col_map(col_map), _row_map(row_map), _nnz(mat.nonZeros())
{
  // Assert overlapping is disabled in the column map
  if (col_map->overlapping())
    throw runtime_error("Ovelapping not supported in this format!");

  _mat_local.reset(new CSRMatrix<T>(mat, exec));
}
//-----------------------------------------------------------------------------
template <typename T>
Matrix<T>::Matrix(const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat_local,
                  const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat_remote,
                  shared_ptr<spmv::L2GMap> col_map,
                  shared_ptr<spmv::L2GMap> row_map,
                  std::shared_ptr<DeviceExecutor> exec)
    : _exec(exec), _col_map(col_map), _row_map(row_map),
      _nnz(mat_local.nonZeros() + mat_remote.nonZeros())
{
  // Assert overlapping is enabled in the column map
  if (!col_map->overlapping())
    throw runtime_error("Ovelapping not enabled in column mapping!");

  _mat_local.reset(new CSRMatrix<T>(mat_local, exec));
  _mat_remote.reset(new CSRMatrix<T>(mat_remote, exec));
}
//-----------------------------------------------------------------------------
template <typename T>
Matrix<T>::Matrix(const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat_local,
                  const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat_remote,
                  const Eigen::Matrix<T, Eigen::Dynamic, 1>& mat_diagonal,
                  shared_ptr<spmv::L2GMap> col_map,
                  shared_ptr<spmv::L2GMap> row_map, int nnz_full,
                  std::shared_ptr<DeviceExecutor> exec)
    : _exec(exec), _col_map(col_map), _row_map(row_map), _nnz(nnz_full),
      _symmetric(true)
{
  _mat_local.reset(new CSRMatrix<T>(mat_local, mat_diagonal, _symmetric, exec));
  _mat_remote.reset(new CSRMatrix<T>(mat_remote, exec));
}
//-----------------------------------------------------------------------------
template <typename T>
Matrix<T>::~Matrix()
{
}
//-----------------------------------------------------------------------------
template <typename T>
int Matrix<T>::rows() const
{
  if (_mat_local != nullptr)
    return _mat_local->rows();
  else if (_mat_remote != nullptr)
    return _mat_remote->rows();
  else
    return 0;
}
//-----------------------------------------------------------------------------
template <typename T>
int Matrix<T>::cols() const
{
  if (_mat_local != nullptr)
    return _mat_local->cols();
  else if (_mat_remote != nullptr)
    return _mat_remote->cols();
  else
    return 0;
}
//-----------------------------------------------------------------------------
template <typename T>
int Matrix<T>::non_zeros() const
{
  if (_symmetric)
    return _nnz;
  else if (_col_map->overlapping())
    return _mat_local->non_zeros() + _mat_remote->non_zeros();
  else
    return _mat_local->non_zeros();
}
//-----------------------------------------------------------------------------
template <typename T>
size_t Matrix<T>::format_size() const
{
  size_t total_bytes;

  total_bytes = sizeof(int) * _mat_local->rows()
                + (sizeof(int) + sizeof(T)) * _mat_local->non_zeros();
  // Contribution of remote block
  if (_symmetric || _col_map->overlapping())
    total_bytes += sizeof(int) * _mat_remote->rows()
                   + (sizeof(int) + sizeof(T)) * _mat_remote->non_zeros();
  // Contribution of diagonal
  if (_symmetric)
    total_bytes += sizeof(T) * _mat_local->rows();

  return total_bytes;
}
//-----------------------------------------------------------------------------
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
Matrix<T>::mult(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const
{
  if (_symmetric && _col_map->overlapping())
    return spmv_sym_overlap(x);
  if (_symmetric)
    return spmv_sym(x);
  if (_col_map->overlapping())
    return spmv_overlap(x);
  return spmv(x);
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::mult(T* x, T* y) const
{
  if (_symmetric && _col_map->overlapping())
    spmv_sym_overlap(x, y);
  else if (_symmetric)
    spmv_sym(x, y);
  else if (_col_map->overlapping())
    spmv_overlap(x, y);
  else
    spmv(x, y);
}
//-----------------------------------------------------------------------------
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
Matrix<T>::transpmult(const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const
{
  throw runtime_error("transpmult() operation not yet implemented");
}
//-----------------------------------------------------------------------------
template <typename T>
Matrix<T>*
Matrix<T>::create_matrix(MPI_Comm comm, std::shared_ptr<DeviceExecutor> exec,
                         const Eigen::SparseMatrix<T, Eigen::RowMajor> mat,
                         int64_t nrows_local, int64_t ncols_local,
                         vector<int64_t> row_ghosts, vector<int64_t> col_ghosts,
                         bool symmetric, CommunicationModel cm)
{
  return create_matrix(comm, exec, mat.outerIndexPtr(), mat.innerIndexPtr(),
                       mat.valuePtr(), nrows_local, ncols_local, row_ghosts,
                       col_ghosts, symmetric, cm);
}
//-----------------------------------------------------------------------------
template <typename T>
Matrix<T>* Matrix<T>::create_matrix(
    MPI_Comm comm, std::shared_ptr<DeviceExecutor> exec, const int32_t* rowptr,
    const int32_t* colind, const T* values, int64_t nrows_local,
    int64_t ncols_local, vector<int64_t> row_ghosts, vector<int64_t> col_ghosts,
    bool symmetric, CommunicationModel cm)
{
  int mpi_size, mpi_rank;
  CHECK_MPI(MPI_Comm_size(comm, &mpi_size));
  CHECK_MPI(MPI_Comm_rank(comm, &mpi_rank));

  vector<int64_t> row_ranges(mpi_size + 1, 0);
  CHECK_MPI(MPI_Allgather(&nrows_local, 1, MPI_INT64_T, row_ranges.data() + 1,
                          1, MPI_INT64_T, comm));
  for (int i = 0; i < mpi_size; ++i)
    row_ranges[i + 1] += row_ranges[i];

  // FIX: often same as rows?
  vector<int64_t> col_ranges(mpi_size + 1, 0);
  CHECK_MPI(MPI_Allgather(&ncols_local, 1, MPI_INT64_T, col_ranges.data() + 1,
                          1, MPI_INT64_T, comm));
  for (int i = 0; i < mpi_size; ++i)
    col_ranges[i + 1] += col_ranges[i];

  // Locate owner process for each row
  vector<int> row_owner(row_ghosts.size());
  for (size_t i = 0; i < row_ghosts.size(); ++i) {
    auto it = upper_bound(row_ranges.begin(), row_ranges.end(), row_ghosts[i]);
    assert(it != row_ranges.end());
    row_owner[i] = it - row_ranges.begin() - 1;
    assert(row_owner[i] != mpi_rank);
  }

  // Create a neighbour comm, remap row_owner to neighbour number
  set<int> neighbour_set(row_owner.begin(), row_owner.end());
  vector<int> dests(neighbour_set.begin(), neighbour_set.end());
  map<int, int> proc_to_dest;
  for (size_t i = 0; i < dests.size(); ++i)
    proc_to_dest.insert({dests[i], i});
  for (auto& q : row_owner)
    q = proc_to_dest[q];

  // Get list of sources (may be different from dests, requires AlltoAll to
  // find)
  vector<char> is_dest(mpi_size, 0);
  for (int d : dests)
    is_dest[d] = 1;
  vector<char> is_source(mpi_size, 0);
  CHECK_MPI(MPI_Alltoall(is_dest.data(), 1, MPI_CHAR, is_source.data(), 1,
                         MPI_CHAR, comm));
  vector<int> sources;
  for (int i = 0; i < mpi_size; ++i)
    if (is_source[i] == 1)
      sources.push_back(i);

  MPI_Comm neighbour_comm;
  CHECK_MPI(MPI_Dist_graph_create_adjacent(
      comm, sources.size(), sources.data(), MPI_UNWEIGHTED, dests.size(),
      dests.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neighbour_comm));

  // send all ghost rows to their owners, using global col idx.
  const int32_t* Aouter = rowptr;
  const int32_t* Ainner = colind;
  const T* Aval = values;

  vector<vector<int64_t>> p_to_index(dests.size());
  vector<vector<T>> p_to_val(dests.size());
  for (size_t i = 0; i < row_ghosts.size(); ++i) {
    const int p = row_owner[i];
    assert(p != -1);
    p_to_index[p].push_back(row_ghosts[i]);
    p_to_val[p].push_back(0.0);
    p_to_index[p].push_back(Aouter[nrows_local + i + 1]
                            - Aouter[nrows_local + i]);
    p_to_val[p].push_back(0.0);

    const int64_t local_offset = col_ranges[mpi_rank];
    for (int j = Aouter[nrows_local + i]; j < Aouter[nrows_local + i + 1];
         ++j) {
      int64_t global_index;
      if (Ainner[j] < ncols_local)
        global_index = Ainner[j] + local_offset;
      else {
        assert(Ainner[j] - ncols_local < (int)col_ghosts.size());
        global_index = col_ghosts[Ainner[j] - ncols_local];
      }
      p_to_index[p].push_back(global_index);
      p_to_val[p].push_back(Aval[j]);
    }
  }

  vector<int> send_size(dests.size());
  vector<int64_t> send_index;
  vector<T> send_val;
  vector<int> send_offset = {0};
  for (size_t p = 0; p < dests.size(); ++p) {
    send_index.insert(send_index.end(), p_to_index[p].begin(),
                      p_to_index[p].end());
    send_val.insert(send_val.end(), p_to_val[p].begin(), p_to_val[p].end());
    assert(p_to_val[p].size() == p_to_index[p].size());
    send_size[p] = p_to_index[p].size();
    send_offset.push_back(send_index.size());
  }

  vector<int> recv_size(sources.size());
  CHECK_MPI(MPI_Neighbor_alltoall(send_size.data(), 1, MPI_INT,
                                  recv_size.data(), 1, MPI_INT,
                                  neighbour_comm));

  vector<int> recv_offset = {0};
  for (int r : recv_size)
    recv_offset.push_back(recv_offset.back() + r);

  vector<int64_t> recv_index(recv_offset.back());
  vector<T> recv_val(recv_offset.back());

  CHECK_MPI(MPI_Neighbor_alltoallv(
      send_index.data(), send_size.data(), send_offset.data(), MPI_INT64_T,
      recv_index.data(), recv_size.data(), recv_offset.data(), MPI_INT64_T,
      neighbour_comm));

  CHECK_MPI(MPI_Neighbor_alltoallv(
      send_val.data(), send_size.data(), send_offset.data(), mpi_type<T>(),
      recv_val.data(), recv_size.data(), recv_offset.data(), mpi_type<T>(),
      neighbour_comm));

  // Create new map from global column index to local
  map<int64_t, int> col_ghost_map;
  for (int64_t q : col_ghosts)
    col_ghost_map.insert({q, -1});

  // Add any new ghost columns
  int pos = 0;
  while (pos < (int)recv_index.size()) {
    //    int64_t global_row = recv_index[pos];
    ++pos;
    int nnz = recv_index[pos];
    ++pos;
    for (int k = 0; k < nnz; ++k) {
      const int64_t recv_col = recv_index[pos];
      ++pos;
      if (recv_col >= col_ranges[mpi_rank + 1]
          or recv_col < col_ranges[mpi_rank])
        col_ghost_map.insert({recv_col, -1});
    }
  }

  // Unique numbering of ghost cols
  int c = ncols_local;
  for (auto& q : col_ghost_map)
    q.second = c++;

  vector<Eigen::Triplet<T>> mat_data;
  vector<Eigen::Triplet<T>> mat_remote_data;
  vector<Eigen::Triplet<T>> mat_diagonal_data;
  for (int row = 0; row < nrows_local; ++row) {
    for (int j = Aouter[row]; j < Aouter[row + 1]; ++j) {
      int col = Ainner[j];
      if (col >= ncols_local) {
        // Get remapped ghost column
        int64_t global_col = col_ghosts[col - ncols_local];
        auto it = col_ghost_map.find(global_col);
        assert(it != col_ghost_map.end());
        col = it->second;
        assert(col >= ncols_local);
      }

      assert(row >= 0 and row < nrows_local);
      assert(col >= 0 and col < (int)(ncols_local + col_ghost_map.size()));
      if (symmetric) {
        // If element is in local column range, insert only if it's on or below
        // main diagonal
        if (col < ncols_local) {
          int64_t global_row = row + row_ranges[mpi_rank];
          int64_t global_col = col + col_ranges[mpi_rank];
          if (global_row > global_col)
            mat_data.push_back(Eigen::Triplet<T>(row, col, Aval[j]));
          else if (global_row == global_col)
            mat_diagonal_data.push_back(Eigen::Triplet<T>(row, col, Aval[j]));
        } else {
          mat_remote_data.push_back(Eigen::Triplet<T>(row, col, Aval[j]));
        }
      } else if (cm == CommunicationModel::p2p_nonblocking
                 || cm == CommunicationModel::collective_nonblocking) {
        if (col < ncols_local)
          mat_data.push_back(Eigen::Triplet<T>(row, col, Aval[j]));
        else
          mat_remote_data.push_back(Eigen::Triplet<T>(row, col, Aval[j]));
      } else {
        mat_data.push_back(Eigen::Triplet<T>(row, col, Aval[j]));
      }
    }
  }

  // Add received data
  pos = 0;
  while (pos < (int)recv_index.size()) {
    int64_t global_row = recv_index[pos];
    assert(global_row >= row_ranges[mpi_rank]
           and global_row < row_ranges[mpi_rank + 1]);
    int32_t row = global_row - row_ranges[mpi_rank];
    ++pos;
    int nnz = recv_index[pos];
    ++pos;
    for (int k = 0; k < nnz; ++k) {
      const int64_t global_col = recv_index[pos];
      const T val = recv_val[pos];
      ++pos;
      int col;
      if (global_col >= col_ranges[mpi_rank + 1]
          or global_col < col_ranges[mpi_rank]) {
        auto it = col_ghost_map.find(global_col);
        assert(it != col_ghost_map.end());
        col = it->second;
      } else
        col = global_col - col_ranges[mpi_rank];
      assert(row >= 0 and row < nrows_local);
      assert(col >= 0 and col < (int)(ncols_local + col_ghost_map.size()));

      if (symmetric) {
        // If element is in local column range, insert only if it's on or below
        // main diagonal
        if (col < ncols_local) {
          if (global_row > global_col)
            mat_data.push_back(Eigen::Triplet<T>(row, col, val));
          else if (global_row == global_col)
            mat_diagonal_data.push_back(Eigen::Triplet<T>(row, col, val));
        } else {
          mat_remote_data.push_back(Eigen::Triplet<T>(row, col, val));
        }
      } else if (cm == CommunicationModel::p2p_nonblocking
                 || cm == CommunicationModel::collective_nonblocking) {
        if (col < ncols_local)
          mat_data.push_back(Eigen::Triplet<T>(row, col, val));
        else
          mat_remote_data.push_back(Eigen::Triplet<T>(row, col, val));
      } else {
        mat_data.push_back(Eigen::Triplet<T>(row, col, val));
      }
    }
  }

  // Rebuild sparse matrix
  vector<int64_t> new_col_ghosts;
  for (auto& q : col_ghost_map)
    new_col_ghosts.push_back(q.first);

  if (symmetric) {
    // Rebuild the sparse matrix block into two sub-blocks
    // The "local" sub-block includes nonzeros in the lower half of the matrix
    // within the local column range of this rank The "remote" sub-block
    // includes all nonzeros out of the local column range of this rank
    auto Blocal = make_shared<Eigen::SparseMatrix<T, Eigen::RowMajor>>(
        nrows_local, ncols_local + new_col_ghosts.size());
    auto Bremote = make_shared<Eigen::SparseMatrix<T, Eigen::RowMajor>>(
        nrows_local, ncols_local + new_col_ghosts.size());
    auto Bdiagonal
        = make_shared<Eigen::Matrix<T, Eigen::Dynamic, 1>>(nrows_local);

    Blocal->setFromTriplets(mat_data.begin(), mat_data.end());
    Bremote->setFromTriplets(mat_remote_data.begin(), mat_remote_data.end());
    Bdiagonal->setZero();
    T* diag = Bdiagonal->data();
    unordered_set<int64_t> unique_rows;
    for (auto const& elem : mat_diagonal_data) {
      unique_rows.insert(elem.row());
      diag[elem.row()] += elem.value();
    }

    shared_ptr<spmv::L2GMap> col_map = make_shared<spmv::L2GMap>(
        comm, ncols_local, new_col_ghosts, exec, cm);
    shared_ptr<spmv::L2GMap> row_map
        = make_shared<spmv::L2GMap>(comm, nrows_local, vector<int64_t>(), exec);

    // Number of nonzeros in full matrix
    int64_t nnz
        = 2 * Blocal->nonZeros() + Bremote->nonZeros() + unique_rows.size();
    return new spmv::Matrix<T>(*Blocal, *Bremote, *Bdiagonal, col_map, row_map,
                               nnz, exec);
  } else if (cm == CommunicationModel::p2p_nonblocking
             || cm == CommunicationModel::collective_nonblocking) {
    // Rebuild the sparse matrix block into two sub-blocks
    // The "local" sub-block includes nonzeros within the local column
    // range of this rank. The "remote" sub-block includes all nonzeros out
    // of the local column range of this rank
    auto Blocal = make_shared<Eigen::SparseMatrix<T, Eigen::RowMajor>>(
        nrows_local, ncols_local);
    auto Bremote = make_shared<Eigen::SparseMatrix<T, Eigen::RowMajor>>(
        nrows_local, ncols_local + new_col_ghosts.size());

    Blocal->setFromTriplets(mat_data.begin(), mat_data.end());
    Bremote->setFromTriplets(mat_remote_data.begin(), mat_remote_data.end());

    shared_ptr<spmv::L2GMap> col_map = make_shared<spmv::L2GMap>(
        comm, ncols_local, new_col_ghosts, exec, cm);
    shared_ptr<spmv::L2GMap> row_map
        = make_shared<spmv::L2GMap>(comm, nrows_local, vector<int64_t>(), exec);

    return new spmv::Matrix<T>(*Blocal, *Bremote, col_map, row_map, exec);
  } else {
    auto B = make_shared<Eigen::SparseMatrix<T, Eigen::RowMajor>>(
        nrows_local, ncols_local + new_col_ghosts.size());

    B->setFromTriplets(mat_data.begin(), mat_data.end());

    shared_ptr<spmv::L2GMap> col_map = make_shared<spmv::L2GMap>(
        comm, ncols_local, new_col_ghosts, exec, cm);
    shared_ptr<spmv::L2GMap> row_map
        = make_shared<spmv::L2GMap>(comm, nrows_local, vector<int64_t>(), exec);

    return new spmv::Matrix<T>(*B, col_map, row_map, exec);
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::spmv(T* x, T* y) const
{
  _mat_local->mult(1, x, 0, y);
}
//-----------------------------------------------------------------------------
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
Matrix<T>::spmv(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> y(_mat_local->rows());
  spmv(x.data(), y.data());
  return y;
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::spmv_overlap(T* x, T* y) const
{
  // Compute SpMV on local block
  _mat_local->mult(1, x, 0, y);

  // Finalise ghost updates
  _col_map->update_finalise(x);

  // Compute SpMV on remote block
  if (_mat_local->non_zeros() > 0)
    _mat_remote->mult(1, x, 1, y);
  else
    _mat_remote->mult(1, x, 0, y);
}
//-----------------------------------------------------------------------------
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
Matrix<T>::spmv_overlap(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> y(_mat_local->rows());
  spmv_overlap(x.data(), y.data());
  return y;
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::spmv_sym(T* x, T* y) const
{
  // Compute symmetric SpMV on local block
  _mat_local->mult(1, x, 0, y);

  // Compute vanilla SpMV on remote block
  _mat_remote->mult(1, x, 1, y);
}
//-----------------------------------------------------------------------------
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
Matrix<T>::spmv_sym(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> y(_mat_local->rows());
  spmv_sym(x.data(), y.data());
  return y;
}
//-----------------------------------------------------------------------------
template <typename T>
void Matrix<T>::spmv_sym_overlap(T* x, T* y) const
{
  // Compute symmetric SpMV on local block
  _mat_local->mult(1, x, 0, y);

  // Finalise ghost updates
  _col_map->update_finalise(x);

  // Compute vanilla SpMV on remote block
  _mat_remote->mult(1, x, 1, y);
}
//-----------------------------------------------------------------------------
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> Matrix<T>::spmv_sym_overlap(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x) const
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> y(_mat_local->rows());
  spmv_sym_overlap(x.data(), y.data());
  return y;
}
//-----------------------------------------------------------------------------

// Explicit instantiation
template class spmv::Matrix<float>;
template class spmv::Matrix<double>;
// template class spmv::Matrix<complex<float>>;
// template class spmv::Matrix<complex<double>>;

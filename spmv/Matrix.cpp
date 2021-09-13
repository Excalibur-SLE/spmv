// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "Matrix.h"
#include "L2GMap.h"
#include "mpi_types.h"
#include <numeric>
#include <set>
#include <unordered_set>

using namespace spmv;

#if defined(_OPENMP) || defined(_SYCL)
//-----------------------------------------------------------------------------
static void** calloc_2d(size_t dim1, size_t dim2, size_t size)
{
  char** ret = (char**)malloc(dim1 * sizeof(char*));
  if (ret != nullptr)
  {
    char* area = (char*)calloc(dim1 * dim2, size);
    if (area != nullptr)
    {
      for (size_t i = 0; i < dim1; ++i)
      {
        ret[i] = (char*)&area[i * dim2 * size];
      }
    }
    else
    {
      free(ret);
      ret = nullptr;
    }
  }

  return (void**)ret;
}
//---------------------
static void free_2d(void** array)
{
  free(array[0]);
  free(array);
}
//---------------------
static size_t get_num_threads()
{
#ifdef _OPENMP
  const char* threads_env = getenv("OMP_NUM_THREADS");
#else // _SYCL
  // FIXME: this is specific to DPC++
  const char* threads_env = getenv("DPCPP_CPU_NUM_CUS");
#endif
  int ret = 1;

  if (threads_env)
  {
    ret = atoi(threads_env);
    if (ret < 0)
      ret = 1;
  }

  return ret;
}
//-----------------------------------------------------------------------------
#endif // _OPENMP || _SYCL

//-----------------------------------------------------------------------------
template <typename T>
Matrix<T>::Matrix(
    std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat,
    std::shared_ptr<spmv::L2GMap> col_map,
    std::shared_ptr<spmv::L2GMap> row_map)
    : _mat_local(mat), _mat_remote(nullptr), _mat_diagonal(nullptr),
      _col_map(col_map), _row_map(row_map), _nnz(mat->nonZeros()),
      _symmetric(false)
#if defined(_OPENMP) || defined(_SYCL)
      ,
      _nthreads(1), _cnfl_map(nullptr), _row_split(nullptr),
      _map_start(nullptr), _map_end(nullptr), _y_local(nullptr)
#endif // _OPENMP
{
  // Assert overlapping is disabled in the column map
  if (col_map->overlapping())
    throw std::runtime_error("Ovelapping not supported in this format!");
#ifdef EIGEN_USE_MKL_ALL
  mkl_init();
#endif // _EIGEN_USE_MKL_ALL

#ifdef _SYCL
  // Initialise SYCL buffers, ownership is passed to SYCL runtime
  // auto property_list
  //     = cl::sycl::property_list{cl::sycl::property::buffer::use_host_ptr()};
  _d_rowptr_local = new sycl::buffer<int>(
      _mat_local->outerIndexPtr(), sycl::range<1>(_mat_local->rows() + 1));
  _d_colind_local = new sycl::buffer<int>(
      _mat_local->innerIndexPtr(), sycl::range<1>(_mat_local->nonZeros()));
  _d_values_local = new sycl::buffer<T>(_mat_local->valuePtr(),
                                        sycl::range<1>(_mat_local->nonZeros()));
  _d_rowptr_remote = nullptr;
  _d_colind_remote = nullptr;
  _d_values_remote = nullptr;
  _d_row_split = nullptr;
  _d_diagonal = nullptr;
  _d_map_start = nullptr;
  _d_map_end = nullptr;
  _d_cnfl_vid = nullptr;
  _d_cnfl_pos = nullptr;
  _d_y_local = nullptr;
#endif // _SYCL
}
//---------------------
template <typename T>
Matrix<T>::Matrix(
    std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat_local,
    std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat_remote,
    std::shared_ptr<spmv::L2GMap> col_map,
    std::shared_ptr<spmv::L2GMap> row_map)
    : _mat_local(mat_local), _mat_remote(mat_remote), _mat_diagonal(nullptr),
      _col_map(col_map), _row_map(row_map),
      _nnz(mat_local->nonZeros() + mat_remote->nonZeros()), _symmetric(false)
#if defined(_OPENMP) || defined(_SYCL)
      ,
      _nthreads(1), _cnfl_map(nullptr), _row_split(nullptr),
      _map_start(nullptr), _map_end(nullptr), _y_local(nullptr)
#endif // _OPENMP
{
  // Assert overlapping is enabled in the column map
  if (!col_map->overlapping())
    throw std::runtime_error("Ovelapping not enabled in column mapping!");

#ifdef EIGEN_USE_MKL_ALL
  mkl_init();
#endif // _EIGEN_USE_MKL_ALL

#ifdef _SYCL
  // Initialise SYCL buffers, ownership is passed to SYCL runtime
  _d_rowptr_local = new sycl::buffer<int>(
      _mat_local->outerIndexPtr(), sycl::range<1>(_mat_local->rows() + 1));
  _d_colind_local = new sycl::buffer<int>(
      _mat_local->innerIndexPtr(), sycl::range<1>(_mat_local->nonZeros()));
  _d_values_local = new sycl::buffer<T>(_mat_local->valuePtr(),
                                        sycl::range<1>(_mat_local->nonZeros()));
  if (_mat_remote->nonZeros() > 0)
  {
    _d_rowptr_remote = new sycl::buffer<int>(
        _mat_remote->outerIndexPtr(), sycl::range<1>(_mat_remote->rows() + 1));
    _d_colind_remote = new sycl::buffer<int>(
        _mat_remote->innerIndexPtr(), sycl::range<1>(_mat_remote->nonZeros()));
    _d_values_remote = new sycl::buffer<T>(
        _mat_remote->valuePtr(), sycl::range<1>(_mat_remote->nonZeros()));
  }
  else
  {
    _d_rowptr_remote = nullptr;
    _d_colind_remote = nullptr;
    _d_values_remote = nullptr;
  }
  _d_row_split = nullptr;
  _d_diagonal = nullptr;
  _d_map_start = nullptr;
  _d_map_end = nullptr;
  _d_cnfl_vid = nullptr;
  _d_cnfl_pos = nullptr;
  _d_y_local = nullptr;
#endif // _SYCL
}
//---------------------
template <typename T>
Matrix<T>::Matrix(
    std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat_local,
    std::shared_ptr<const Eigen::SparseMatrix<T, Eigen::RowMajor>> mat_remote,
    std::shared_ptr<const Eigen::Matrix<T, Eigen::Dynamic, 1>> mat_diagonal,
    std::shared_ptr<spmv::L2GMap> col_map,
    std::shared_ptr<spmv::L2GMap> row_map, int nnz_full)
    : _mat_local(mat_local), _mat_remote(mat_remote),
      _mat_diagonal(mat_diagonal), _col_map(col_map), _row_map(row_map),
      _nnz(nnz_full), _symmetric(true)
#if defined(_OPENMP) || defined(_SYCL)
      ,
      _nthreads(1), _cnfl_map(nullptr), _row_split(nullptr),
      _map_start(nullptr), _map_end(nullptr), _y_local(nullptr)
#endif // _OPENMP
{
#if defined(_OPENMP) || defined(_SYCL)
  _nthreads = get_num_threads();
  tune(_nthreads);
#endif // _OPENMP

#ifdef _SYCL
  // Initialise SYCL buffers, ownership is passed to SYCL runtime
  _d_rowptr_local = new sycl::buffer<int>(
      _mat_local->outerIndexPtr(), sycl::range<1>(_mat_local->rows() + 1));
  _d_colind_local = new sycl::buffer<int>(
      _mat_local->innerIndexPtr(), sycl::range<1>(_mat_local->nonZeros()));
  _d_values_local = new sycl::buffer<T>(_mat_local->valuePtr(),
                                        sycl::range<1>(_mat_local->nonZeros()));
  if (_mat_remote->nonZeros() > 0)
  {
    _d_rowptr_remote = new sycl::buffer<int>(
        _mat_remote->outerIndexPtr(), sycl::range<1>(_mat_remote->rows() + 1));
    _d_colind_remote = new sycl::buffer<int>(
        _mat_remote->innerIndexPtr(), sycl::range<1>(_mat_remote->nonZeros()));
    _d_values_remote = new sycl::buffer<T>(
        _mat_remote->valuePtr(), sycl::range<1>(_mat_remote->nonZeros()));
  }
  else
  {
    _d_rowptr_remote = nullptr;
    _d_colind_remote = nullptr;
    _d_values_remote = nullptr;
  }
  _d_diagonal = new sycl::buffer<T>(_mat_diagonal->data(),
                                    sycl::range<1>(_mat_local->rows()));
  _d_row_split
      = new sycl::buffer<int>(_row_split, sycl::range<1>(_nthreads + 1));
  _d_map_start = new sycl::buffer<int>(
      _map_start, sycl::range<1>{static_cast<size_t>(_nthreads)});
  _d_map_end = new sycl::buffer<int>(
      _map_end, sycl::range<1>{static_cast<size_t>(_nthreads)});
  _d_cnfl_vid = new sycl::buffer<short>(
      _cnfl_map->vid, sycl::range<1>{static_cast<size_t>(_ncnfls)});
  _d_cnfl_pos = new sycl::buffer<int>(
      _cnfl_map->pos, sycl::range<1>{static_cast<size_t>(_ncnfls)});
  _d_y_local = new sycl::buffer<T, 2>(
      &_y_local[0][0], sycl::range<2>{static_cast<size_t>(_nthreads),
                                      static_cast<size_t>(_mat_local->rows())});
#endif // _SYCL
}
//---------------------
template <typename T>
Matrix<T>::~Matrix()
{
#ifdef _SYCL
  // SYCL buffers to return ownership of data
  delete _d_rowptr_local;
  delete _d_colind_local;
  delete _d_values_local;
  delete _d_rowptr_remote;
  delete _d_colind_remote;
  delete _d_values_remote;
  if (_symmetric)
  {
    delete _d_row_split;
    delete _d_diagonal;
    delete _d_map_start;
    delete _d_map_end;
    delete _d_cnfl_vid;
    delete _d_cnfl_pos;
    delete _d_y_local;
  }
#endif // _SYCL

#ifdef EIGEN_USE_MKL_ALL
  if (!_symmetric)
  {
    mkl_sparse_destroy(_mat_local_mkl);
  }
  if (_col_map->overlapping())
    mkl_sparse_destroy(_mat_remote_mkl);
#endif // _EIGEN_USE_MKL_ALL

#if defined(_OPENMP) || defined(_SYCL)
  if (_symmetric)
  {
    delete _cnfl_map;
    delete[] _row_split;
    delete[] _map_start;
    delete[] _map_end;
    free_2d((void**)_y_local);
  }
#endif // _OPENMP || _SYCL
}
//---------------------
template <typename T>
int Matrix<T>::non_zeros() const
{
  if (_symmetric)
    return _nnz;
  else if (_col_map->overlapping())
    return _mat_local->nonZeros() + _mat_remote->nonZeros();
  else
    return _mat_local->nonZeros();
}
//---------------------
template <typename T>
size_t Matrix<T>::format_size() const
{
  size_t total_bytes;

  total_bytes = sizeof(int) * _mat_local->rows()
                + (sizeof(int) + sizeof(T)) * _mat_local->nonZeros();
  // Contribution of remote block
  if (_symmetric || _col_map->overlapping())
    total_bytes += sizeof(int) * _mat_remote->rows()
                   + (sizeof(int) + sizeof(T)) * _mat_remote->nonZeros();
  // Contribution of diagonal
  if (_symmetric)
    total_bytes += sizeof(T) * _mat_local->rows();

  return total_bytes;
}
//---------------------
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> Matrix<T>::
operator*(Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const
{
  if (_symmetric && _col_map->overlapping())
    return spmv_sym_overlap(b);
  if (_symmetric)
    return spmv_sym(b);
  if (_col_map->overlapping())
    return spmv_overlap(b);
  return (*_mat_local) * b;
}
//---------------------
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
Matrix<T>::transpmult(const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const
{
  if (_symmetric)
    throw std::runtime_error(
        "transpmult() operation not yet implemented for symmetric matrices");
  else if (_col_map->overlapping())
    return _mat_local->transpose() * b + _mat_remote->transpose() * b;
  else
    return _mat_local->transpose() * b;
}
//---------------------
template <typename T>
Matrix<T>* Matrix<T>::create_matrix(
    MPI_Comm comm, const Eigen::SparseMatrix<T, Eigen::RowMajor> mat,
    std::int64_t nrows_local, std::int64_t ncols_local,
    std::vector<std::int64_t> row_ghosts, std::vector<std::int64_t> col_ghosts,
    bool symmetric, CommunicationModel cm)
{
  return create_matrix(comm, mat.outerIndexPtr(), mat.innerIndexPtr(),
                       mat.valuePtr(), nrows_local, ncols_local, row_ghosts,
                       col_ghosts, symmetric, cm);
}
//---------------------
template <typename T>
Matrix<T>* Matrix<T>::create_matrix(MPI_Comm comm, const std::int32_t* rowptr,
                                    const std::int32_t* colind, const T* values,
                                    std::int64_t nrows_local,
                                    std::int64_t ncols_local,
                                    std::vector<std::int64_t> row_ghosts,
                                    std::vector<std::int64_t> col_ghosts,
                                    bool symmetric, CommunicationModel cm)
{
  int mpi_size, mpi_rank;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &mpi_rank);

  std::vector<std::int64_t> row_ranges(mpi_size + 1, 0);
  MPI_Allgather(&nrows_local, 1, MPI_INT64_T, row_ranges.data() + 1, 1,
                MPI_INT64_T, comm);
  for (int i = 0; i < mpi_size; ++i)
    row_ranges[i + 1] += row_ranges[i];

  // FIX: often same as rows?
  std::vector<std::int64_t> col_ranges(mpi_size + 1, 0);
  MPI_Allgather(&ncols_local, 1, MPI_INT64_T, col_ranges.data() + 1, 1,
                MPI_INT64_T, comm);
  for (int i = 0; i < mpi_size; ++i)
    col_ranges[i + 1] += col_ranges[i];

  // Locate owner process for each row
  std::vector<int> row_owner(row_ghosts.size());
  for (std::size_t i = 0; i < row_ghosts.size(); ++i)
  {
    auto it
        = std::upper_bound(row_ranges.begin(), row_ranges.end(), row_ghosts[i]);
    assert(it != row_ranges.end());
    row_owner[i] = it - row_ranges.begin() - 1;
    assert(row_owner[i] != mpi_rank);
  }

  // Create a neighbour comm, remap row_owner to neighbour number
  std::set<int> neighbour_set(row_owner.begin(), row_owner.end());
  std::vector<int> dests(neighbour_set.begin(), neighbour_set.end());
  std::map<int, int> proc_to_dest;
  for (std::size_t i = 0; i < dests.size(); ++i)
    proc_to_dest.insert({dests[i], i});
  for (auto& q : row_owner)
    q = proc_to_dest[q];

  // Get list of sources (may be different from dests, requires AlltoAll to
  // find)
  std::vector<char> is_dest(mpi_size, 0);
  for (int d : dests)
    is_dest[d] = 1;
  std::vector<char> is_source(mpi_size, 0);
  MPI_Alltoall(is_dest.data(), 1, MPI_CHAR, is_source.data(), 1, MPI_CHAR,
               comm);
  std::vector<int> sources;
  for (int i = 0; i < mpi_size; ++i)
    if (is_source[i] == 1)
      sources.push_back(i);

  MPI_Comm neighbour_comm;
  MPI_Dist_graph_create_adjacent(
      comm, sources.size(), sources.data(), MPI_UNWEIGHTED, dests.size(),
      dests.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neighbour_comm);

  // send all ghost rows to their owners, using global col idx.
  const std::int32_t* Aouter = rowptr;
  const std::int32_t* Ainner = colind;
  const T* Aval = values;

  std::vector<std::vector<std::int64_t>> p_to_index(dests.size());
  std::vector<std::vector<T>> p_to_val(dests.size());
  for (std::size_t i = 0; i < row_ghosts.size(); ++i)
  {
    const int p = row_owner[i];
    assert(p != -1);
    p_to_index[p].push_back(row_ghosts[i]);
    p_to_val[p].push_back(0.0);
    p_to_index[p].push_back(Aouter[nrows_local + i + 1]
                            - Aouter[nrows_local + i]);
    p_to_val[p].push_back(0.0);

    const std::int64_t local_offset = col_ranges[mpi_rank];
    for (int j = Aouter[nrows_local + i]; j < Aouter[nrows_local + i + 1]; ++j)
    {
      std::int64_t global_index;
      if (Ainner[j] < ncols_local)
        global_index = Ainner[j] + local_offset;
      else
      {
        assert(Ainner[j] - ncols_local < (int)col_ghosts.size());
        global_index = col_ghosts[Ainner[j] - ncols_local];
      }
      p_to_index[p].push_back(global_index);
      p_to_val[p].push_back(Aval[j]);
    }
  }

  std::vector<int> send_size(dests.size());
  std::vector<std::int64_t> send_index;
  std::vector<T> send_val;
  std::vector<int> send_offset = {0};
  for (std::size_t p = 0; p < dests.size(); ++p)
  {
    send_index.insert(send_index.end(), p_to_index[p].begin(),
                      p_to_index[p].end());
    send_val.insert(send_val.end(), p_to_val[p].begin(), p_to_val[p].end());
    assert(p_to_val[p].size() == p_to_index[p].size());
    send_size[p] = p_to_index[p].size();
    send_offset.push_back(send_index.size());
  }

  std::vector<int> recv_size(sources.size());
  MPI_Neighbor_alltoall(send_size.data(), 1, MPI_INT, recv_size.data(), 1,
                        MPI_INT, neighbour_comm);

  std::vector<int> recv_offset = {0};
  for (int r : recv_size)
    recv_offset.push_back(recv_offset.back() + r);

  std::vector<std::int64_t> recv_index(recv_offset.back());
  std::vector<T> recv_val(recv_offset.back());

  MPI_Neighbor_alltoallv(send_index.data(), send_size.data(),
                         send_offset.data(), MPI_INT64_T, recv_index.data(),
                         recv_size.data(), recv_offset.data(), MPI_INT64_T,
                         neighbour_comm);

  MPI_Neighbor_alltoallv(send_val.data(), send_size.data(), send_offset.data(),
                         mpi_type<T>(), recv_val.data(), recv_size.data(),
                         recv_offset.data(), mpi_type<T>(), neighbour_comm);

  // Create new map from global column index to local
  std::map<std::int64_t, int> col_ghost_map;
  for (std::int64_t q : col_ghosts)
    col_ghost_map.insert({q, -1});

  // Add any new ghost columns
  int pos = 0;
  while (pos < (int)recv_index.size())
  {
    //    std::int64_t global_row = recv_index[pos];
    ++pos;
    int nnz = recv_index[pos];
    ++pos;
    for (int k = 0; k < nnz; ++k)
    {
      const std::int64_t recv_col = recv_index[pos];
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

  std::vector<Eigen::Triplet<T>> mat_data;
  std::vector<Eigen::Triplet<T>> mat_remote_data;
  std::vector<Eigen::Triplet<T>> mat_diagonal_data;
  for (int row = 0; row < nrows_local; ++row)
  {
    for (int j = Aouter[row]; j < Aouter[row + 1]; ++j)
    {
      int col = Ainner[j];
      if (col >= ncols_local)
      {
        // Get remapped ghost column
        std::int64_t global_col = col_ghosts[col - ncols_local];
        auto it = col_ghost_map.find(global_col);
        assert(it != col_ghost_map.end());
        col = it->second;
        assert(col >= ncols_local);
      }

      assert(row >= 0 and row < nrows_local);
      assert(col >= 0 and col < (int)(ncols_local + col_ghost_map.size()));
      if (symmetric)
      {
        // If element is in local column range, insert only if it's on or below
        // main diagonal
        if (col < ncols_local)
        {
          std::int64_t global_row = row + row_ranges[mpi_rank];
          std::int64_t global_col = col + col_ranges[mpi_rank];
          if (global_row > global_col)
            mat_data.push_back(Eigen::Triplet<T>(row, col, Aval[j]));
          else if (global_row == global_col)
            mat_diagonal_data.push_back(Eigen::Triplet<T>(row, col, Aval[j]));
        }
        else
        {
          mat_remote_data.push_back(Eigen::Triplet<T>(row, col, Aval[j]));
        }
      }
      else if (cm == CommunicationModel::p2p_nonblocking
               || cm == CommunicationModel::collective_nonblocking)
      {
        if (col < ncols_local)
          mat_data.push_back(Eigen::Triplet<T>(row, col, Aval[j]));
        else
          mat_remote_data.push_back(Eigen::Triplet<T>(row, col, Aval[j]));
      }
      else
      {
        mat_data.push_back(Eigen::Triplet<T>(row, col, Aval[j]));
      }
    }
  }

  // Add received data
  pos = 0;
  while (pos < (int)recv_index.size())
  {
    std::int64_t global_row = recv_index[pos];
    assert(global_row >= row_ranges[mpi_rank]
           and global_row < row_ranges[mpi_rank + 1]);
    std::int32_t row = global_row - row_ranges[mpi_rank];
    ++pos;
    int nnz = recv_index[pos];
    ++pos;
    for (int k = 0; k < nnz; ++k)
    {
      const std::int64_t global_col = recv_index[pos];
      const T val = recv_val[pos];
      ++pos;
      int col;
      if (global_col >= col_ranges[mpi_rank + 1]
          or global_col < col_ranges[mpi_rank])
      {
        auto it = col_ghost_map.find(global_col);
        assert(it != col_ghost_map.end());
        col = it->second;
      }
      else
        col = global_col - col_ranges[mpi_rank];
      assert(row >= 0 and row < nrows_local);
      assert(col >= 0 and col < (int)(ncols_local + col_ghost_map.size()));

      if (symmetric)
      {
        // If element is in local column range, insert only if it's on or below
        // main diagonal
        if (col < ncols_local)
        {
          if (global_row > global_col)
            mat_data.push_back(Eigen::Triplet<T>(row, col, val));
          else if (global_row == global_col)
            mat_diagonal_data.push_back(Eigen::Triplet<T>(row, col, val));
        }
        else
        {
          mat_remote_data.push_back(Eigen::Triplet<T>(row, col, val));
        }
      }
      else if (cm == CommunicationModel::p2p_nonblocking
               || cm == CommunicationModel::collective_nonblocking)
      {
        if (col < ncols_local)
          mat_data.push_back(Eigen::Triplet<T>(row, col, val));
        else
          mat_remote_data.push_back(Eigen::Triplet<T>(row, col, val));
      }
      else
      {
        mat_data.push_back(Eigen::Triplet<T>(row, col, val));
      }
    }
  }

  // Rebuild sparse matrix
  std::vector<std::int64_t> new_col_ghosts;
  for (auto& q : col_ghost_map)
    new_col_ghosts.push_back(q.first);

  if (symmetric)
  {
    // Rebuild the sparse matrix block into two sub-blocks
    // The "local" sub-block includes nonzeros in the lower half of the matrix
    // within the local column range of this rank The "remote" sub-block
    // includes all nonzeros out of the local column range of this rank
    auto Blocal = std::make_shared<Eigen::SparseMatrix<T, Eigen::RowMajor>>(
        nrows_local, ncols_local + new_col_ghosts.size());
    auto Bremote = std::make_shared<Eigen::SparseMatrix<T, Eigen::RowMajor>>(
        nrows_local, ncols_local + new_col_ghosts.size());
    auto Bdiagonal
        = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, 1>>(nrows_local);

    Blocal->setFromTriplets(mat_data.begin(), mat_data.end());
    Bremote->setFromTriplets(mat_remote_data.begin(), mat_remote_data.end());
    Bdiagonal->setZero();
    T* diag = Bdiagonal->data();
    std::unordered_set<std::int64_t> unique_rows;
    for (auto const& elem : mat_diagonal_data)
    {
      unique_rows.insert(elem.row());
      diag[elem.row()] += elem.value();
    }

    std::shared_ptr<spmv::L2GMap> col_map
        = std::make_shared<spmv::L2GMap>(comm, ncols_local, new_col_ghosts, cm);
    std::shared_ptr<spmv::L2GMap> row_map = std::make_shared<spmv::L2GMap>(
        comm, nrows_local, std::vector<std::int64_t>());

    // Number of nonzeros in full matrix
    std::int64_t nnz
        = 2 * Blocal->nonZeros() + Bremote->nonZeros() + unique_rows.size();

    return new spmv::Matrix<T>(Blocal, Bremote, Bdiagonal, col_map, row_map,
                               nnz);
  }
  else if (cm == CommunicationModel::p2p_nonblocking
           || cm == CommunicationModel::collective_nonblocking)
  {
    // Rebuild the sparse matrix block into two sub-blocks
    // The "local" sub-block includes nonzeros within the local column
    // range of this rank. The "remote" sub-block includes all nonzeros out
    // of the local column range of this rank
    auto Blocal = std::make_shared<Eigen::SparseMatrix<T, Eigen::RowMajor>>(
        nrows_local, ncols_local + new_col_ghosts.size());
    auto Bremote = std::make_shared<Eigen::SparseMatrix<T, Eigen::RowMajor>>(
        nrows_local, ncols_local + new_col_ghosts.size());

    Blocal->setFromTriplets(mat_data.begin(), mat_data.end());
    Bremote->setFromTriplets(mat_remote_data.begin(), mat_remote_data.end());

    std::shared_ptr<spmv::L2GMap> col_map
        = std::make_shared<spmv::L2GMap>(comm, ncols_local, new_col_ghosts, cm);
    std::shared_ptr<spmv::L2GMap> row_map = std::make_shared<spmv::L2GMap>(
        comm, nrows_local, std::vector<std::int64_t>());

    return new spmv::Matrix<T>(Blocal, Bremote, col_map, row_map);
  }
  else
  {
    auto B = std::make_shared<Eigen::SparseMatrix<T, Eigen::RowMajor>>(
        nrows_local, ncols_local + new_col_ghosts.size());

    B->setFromTriplets(mat_data.begin(), mat_data.end());

    std::shared_ptr<spmv::L2GMap> col_map
        = std::make_shared<spmv::L2GMap>(comm, ncols_local, new_col_ghosts, cm);
    std::shared_ptr<spmv::L2GMap> row_map = std::make_shared<spmv::L2GMap>(
        comm, nrows_local, std::vector<std::int64_t>());

    return new spmv::Matrix<T>(B, col_map, row_map);
  }
}
//-----------------------------------------------------------------------------
#if defined(_OPENMP) || defined(_SYCL)
template <typename T>
void Matrix<T>::partition_by_nrows(const int nthreads)
{
  if (!_row_split)
  {
    _row_split = new int[nthreads + 1];
  }

  int nrows = _mat_local->rows();
  if (nthreads == 1)
  {
    _row_split[0] = 0;
    _row_split[1] = nrows;
    return;
  }

  // Compute new matrix splits
  int nrows_per_split = (nrows + nthreads - 1) / nthreads;
  int i;
  _row_split[0] = 0;
  for (i = 0; i < nthreads; i++)
  {
    if (_row_split[i] + nrows_per_split < nrows)
    {
      _row_split[i + 1] = _row_split[i] + nrows_per_split;
    }
    else
    {
      _row_split[i + 1] = _row_split[i] + nrows - i * nrows_per_split;
      break;
    }
  }

  for (int j = i; j <= nthreads; j++)
  {
    _row_split[j] = nrows;
  }
}
//---------------------
template <typename T>
void Matrix<T>::partition_by_nnz(const int nthreads)
{
  const int nrows = _mat_local->rows();
  const int nnz = _mat_local->nonZeros() + _mat_remote->nonZeros();
  const int* rowptr = _mat_local->outerIndexPtr();
  const int* rowptr_outer = _mat_remote->outerIndexPtr();

  if (!_row_split)
  {
    _row_split = new int[nthreads + 1];
  }

  if (nthreads == 1)
  {
    _row_split[0] = 0;
    _row_split[1] = nrows;
    return;
  }

  // Compute the matrix splits.
  int nnz_per_split = (nnz + nthreads - 1) / nthreads;
  int curr_nnz = 0;
  int row_start = 0;
  int split_cnt = 0;
  int i;

  _row_split[0] = row_start;
  for (i = 0; i < nrows; i++)
  {
    curr_nnz
        += rowptr[i + 1] - rowptr[i] + rowptr_outer[i + 1] - rowptr_outer[i];
    if (curr_nnz >= nnz_per_split)
    {
      row_start = i + 1;
      ++split_cnt;
      if (split_cnt <= nthreads)
        _row_split[split_cnt] = row_start;
      curr_nnz = 0;
    }
  }

  // Fill the last split with remaining elements
  if (curr_nnz < nnz_per_split && split_cnt <= nthreads)
  {
    _row_split[++split_cnt] = nrows;
  }

  // If there are any remaining rows merge them in last partition
  if (split_cnt > nthreads)
  {
    _row_split[nthreads] = nrows;
  }

  // If there are remaining threads create empty partitions
  for (int i = split_cnt + 1; i <= nthreads; i++)
  {
    _row_split[i] = nrows;
  }
}
//---------------------
template <typename T>
void Matrix<T>::tune(const int nthreads)
{
  // partition_by_nrows(nthreads);
  partition_by_nnz(nthreads);

  if (_symmetric)
  {
    // Allocate buffers for "local vectors indexing" method
    // The first thread writes directly to the output vector, so doesn't need a
    // buffer
#ifdef _OPENMP
    _y_local = (T**)calloc_2d(nthreads - 1, _mat_local->rows(), sizeof(T));
#else  // _SYCL
    _y_local = (T**)calloc_2d(nthreads, _mat_local->rows(), sizeof(T));
#endif // _SYCL

    // Build conflict map for local block
    std::map<int, std::unordered_set<int>> row_conflicts;
    std::set<int> thread_conflicts;
    int ncnfls = 0;
    const int* rowptr = _mat_local->outerIndexPtr();
    const int* colind = _mat_local->innerIndexPtr();
    for (int tid = 1; tid < nthreads; ++tid)
    {
      for (int i = _row_split[tid]; i < _row_split[tid + 1]; ++i)
      {
        for (int j = rowptr[i]; j < rowptr[i + 1]; ++j)
        {
          int target_row = colind[j];
          if (target_row < _row_split[tid])
          {
            thread_conflicts.insert(target_row);
            row_conflicts[target_row].insert(tid);
          }
        }
      }
      ncnfls += thread_conflicts.size();
      thread_conflicts.clear();
    }

    // Finalise conflict map data structure
    _cnfl_map = new ConflictMap(ncnfls);
    int cnt = 0;
    for (auto& conflict : row_conflicts)
    {
      for (auto tid : conflict.second)
      {
        _cnfl_map->pos[cnt] = conflict.first;
        _cnfl_map->vid[cnt] = tid;
        cnt++;
      }
    }
    assert(cnt == ncnfls);

    // Split reduction work among threads so that conflicts to the same row are
    // assigned to the same thread
    _map_start = new int[nthreads]();
    _map_end = new int[nthreads]();
    int total_count = ncnfls;
    int tid = 0;
    int limit = total_count / nthreads;
    int tmp_count = 0, run_cnt = 0;
    for (auto& elem : row_conflicts)
    {
      run_cnt += elem.second.size();
      if (tmp_count < limit)
      {
        tmp_count += elem.second.size();
      }
      else
      {
        _map_end[tid] = tmp_count;
        // If we have exceeded the number of threads, assigned what is left to
        // last thread
        total_count -= tmp_count;
        tmp_count = elem.second.size();
        limit = total_count / (nthreads - (tid + 1));
        tid++;
        if (tid == nthreads - 1)
        {
          break;
        }
      }
    }

    for (int i = tid; i < nthreads; i++)
      _map_end[i] = ncnfls - (run_cnt - tmp_count);

    int start = 0;
    for (int tid = 0; tid < nthreads; tid++)
    {
      _map_start[tid] = start;
      _map_end[tid] += start;
      start = _map_end[tid];
    }

    _ncnfls = ncnfls;
  }
}
#endif // _OPENMP
//---------------------
#ifdef _SYCL
template <typename T>
sycl::event Matrix<T>::spmv_sycl(sycl::queue& q, T* __restrict__ b,
                                 T* __restrict__ y) const
{
#ifdef _MKL
#else
  namespace acc = sycl::access;
  sycl::event event = q.submit([&](sycl::handler& h) {
    const size_t nrows = _mat_local->rows();
    auto rowptr = _d_rowptr_local->template get_access<acc::mode::read>(h);
    auto colind = _d_colind_local->template get_access<acc::mode::read>(h);
    auto values = _d_values_local->template get_access<acc::mode::read>(h);

    h.parallel_for<class spmv>(sycl::range<1>{nrows}, [=](sycl::id<1> it) {
      const int i = it[0];
      T y_tmp = 0;

      for (int j = rowptr[i]; j < rowptr[i + 1]; ++j)
      {
        y_tmp += values[j] * b[colind[j]];
      }

      y[i] = y_tmp;
    });
  });
  return event;
#endif
}
//---------------------
template <typename T>
sycl::event Matrix<T>::spmv_sym_sycl(sycl::queue& q, T* __restrict__ b,
                                     T* __restrict__ y) const
{
  namespace acc = sycl::access;
  sycl::event event;

  // Compute diagonal contribution
  event = q.submit([&](sycl::handler& h) {
    const size_t nrows = _mat_local->rows();
    auto diagonal = _d_diagonal->template get_access<acc::mode::read>(h);

    h.parallel_for<class sym_diagonal>(sycl::range<1>{nrows},
                                       [=](sycl::id<1> it) {
                                         const int i = it[0];
                                         y[i] = diagonal[i] * b[i];
                                       });
  });
  q.wait();

  // Compute symmetric SpMV on local - local vectors phase
  if (_mat_local->nonZeros() > 0)
  {
    event = q.submit([&](sycl::handler& h) {
      auto row_split = _d_row_split->template get_access<acc::mode::read>(h);
      auto rowptr = _d_rowptr_local->template get_access<acc::mode::read>(h);
      auto colind = _d_colind_local->template get_access<acc::mode::read>(h);
      auto values = _d_values_local->template get_access<acc::mode::read>(h);
      auto y_local = _d_y_local->template get_access<acc::mode::read_write>(h);

      h.parallel_for<class sym_lower1>(
          sycl::range<1>{_nthreads}, [=](sycl::id<1> it) {
            const int tid = it[0];
            const int row_offset = row_split[tid];
            for (int i = row_split[tid]; i < row_split[tid + 1]; ++i)
            {
              T y_tmp = 0;

              // Compute symmetric SpMV on local block - local vectors phase
              for (int j = rowptr[i]; j < rowptr[i + 1]; ++j)
              {
                int col = colind[j];
                T val = values[j];
                y_tmp += val * b[col];
                if (col < row_offset)
                {
                  y_local[tid][col] += val * b[i];
                }
                else
                {
                  y[col] += val * b[i];
                }
              }

              y[i] += y_tmp;
            }
          });
    });
  }
  q.wait();

  // Compute symmetric SpMV on remote block - local vectors phase
  if (_mat_remote->nonZeros() > 0)
  {
    event = q.submit([&](sycl::handler& h) {
      auto row_split = _d_row_split->template get_access<acc::mode::read>(h);
      auto rowptr = _d_rowptr_remote->template get_access<acc::mode::read>(h);
      auto colind = _d_colind_remote->template get_access<acc::mode::read>(h);
      auto values = _d_values_remote->template get_access<acc::mode::read>(h);

      h.parallel_for<class sym_lower1>(
          sycl::range<1>{_nthreads}, [=](sycl::id<1> it) {
            const int tid = it[0];
            for (int i = row_split[tid]; i < row_split[tid + 1]; ++i)
            {
              T y_tmp = 0;

              // Compute vanilla SpMV on remote block
              for (int j = rowptr[i]; j < rowptr[i + 1]; ++j)
              {
                y_tmp += values[j] * b[colind[j]];
              }

              y[i] += y_tmp;
            }
          });
    });
  }
  q.wait();

  // Reduction of local vectors phase
  if (_ncnfls > 0)
  {
    event = q.submit([&](sycl::handler& h) {
      auto map_start = _d_map_start->template get_access<acc::mode::read>(h);
      auto map_end = _d_map_end->template get_access<acc::mode::read>(h);
      auto cnfl_vid = _d_cnfl_vid->template get_access<acc::mode::read>(h);
      auto cnfl_pos = _d_cnfl_pos->template get_access<acc::mode::read>(h);
      auto y_local = _d_y_local->template get_access<acc::mode::read_write>(h);

      h.parallel_for<class sym_reduction>(
          sycl::range<1>{_nthreads}, [=](sycl::id<1> it) {
            const int tid = it[0];
            for (int i = map_start[tid]; i < map_end[tid]; ++i)
            {
              int vid = cnfl_vid[i];
              int pos = cnfl_pos[i];
              y[pos] += y_local[vid][pos];
              y_local[vid][pos] = 0.0;
            }
          });
    });
  }

  return event;
}
#endif // _SYCL
//---------------------
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
    Matrix<T>::spmv_overlap(Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> y(_mat_local->rows());
  const int* rowptr = _mat_local->outerIndexPtr();
  const int* colind = _mat_local->innerIndexPtr();
  const T* values = _mat_local->valuePtr();
  const int* rowptr_remote = _mat_remote->outerIndexPtr();
  const int* colind_remote = _mat_remote->innerIndexPtr();
  const T* values_remote = _mat_remote->valuePtr();
  T* b_ptr = b.data();
  T* y_ptr = y.data();

  // Compute SpMV on local block
  for (int i = 0; i < _mat_local->rows(); ++i)
  {
    T y_tmp = 0;

    for (int j = rowptr[i]; j < rowptr[i + 1]; ++j)
    {
      y_tmp += values[j] * b_ptr[colind[j]];
    }

    y_ptr[i] = y_tmp;
  }

  // Finalise ghost updates
  _col_map->update_finalise(b_ptr);

  // Compute SpMV on remote block
  for (int i = 0; i < _mat_remote->rows(); ++i)
  {
    T y_tmp = 0;

    for (int j = rowptr_remote[i]; j < rowptr_remote[i + 1]; ++j)
    {
      y_tmp += values_remote[j] * b_ptr[colind_remote[j]];
    }

    y_ptr[i] += y_tmp;
  }

  return y;
}
//---------------------
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
Matrix<T>::spmv_sym(const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> y(_mat_local->rows());
  const int* rowptr = _mat_local->outerIndexPtr();
  const int* colind = _mat_local->innerIndexPtr();
  const T* values = _mat_local->valuePtr();
  const int* rowptr_remote = _mat_remote->outerIndexPtr();
  const int* colind_remote = _mat_remote->innerIndexPtr();
  const T* values_remote = _mat_remote->valuePtr();
  const T* diagonal = _mat_diagonal->data();
  const T* b_ptr = b.data();
  T* y_ptr = y.data();

#ifdef _OPENMP
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int row_offset = _row_split[tid];
    T* y_local = (tid == 0) ? y_ptr : _y_local[tid - 1];

    // Compute diagonal
    for (int i = _row_split[tid]; i < _row_split[tid + 1]; ++i)
      y_ptr[i] = diagonal[i] * b_ptr[i];
    #pragma omp barrier

    for (int i = _row_split[tid]; i < _row_split[tid + 1]; ++i)
    {
      T y_tmp = 0;

      // Compute symmetric SpMV on local block - local vectors phase
      for (int j = rowptr[i]; j < rowptr[i + 1]; ++j)
      {
        int col = colind[j];
        T val = values[j];
        y_tmp += val * b_ptr[col];
        if (col < row_offset)
        {
          y_local[col] += val * b_ptr[i];
        }
        else
        {
          y_ptr[col] += val * b_ptr[i];
        }
      }

      // Compute vanilla SpMV on remote block
      for (int j = rowptr_remote[i]; j < rowptr_remote[i + 1]; ++j)
      {
        y_tmp += values_remote[j] * b_ptr[colind_remote[j]];
      }

      y_ptr[i] += y_tmp;
    }
    #pragma omp barrier

    // Compute symmetric SpMV on local block - reduction of conflicts phase
    for (int i = _map_start[tid]; i < _map_end[tid]; ++i)
    {
      int vid = _cnfl_map->vid[i];
      int pos = _cnfl_map->pos[i];
      y_ptr[pos] += _y_local[vid - 1][pos];
      _y_local[vid - 1][pos] = 0.0;
    }
  }
#else
  for (int i = 0; i < _mat_local->rows(); ++i)
  {
    T y_tmp = diagonal[i] * b_ptr[i];

    // Compute symmetric SpMV on local block
    for (int j = rowptr[i]; j < rowptr[i + 1]; ++j)
    {
      int col = colind[j];
      T val = values[j];
      y_tmp += val * b_ptr[col];
      y_ptr[col] += val * b_ptr[i];
    }

    // Compute vanilla SpMV on remote block
    for (int j = rowptr_remote[i]; j < rowptr_remote[i + 1]; ++j)
    {
      y_tmp += values_remote[j] * b_ptr[colind_remote[j]];
    }

    y_ptr[i] = y_tmp;
  }
#endif // _OPENMP

  return y;
}
//---------------------
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
    Matrix<T>::spmv_sym_overlap(Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> y(_mat_local->rows());
  const int* rowptr = _mat_local->outerIndexPtr();
  const int* colind = _mat_local->innerIndexPtr();
  const T* values = _mat_local->valuePtr();
  const int* rowptr_remote = _mat_remote->outerIndexPtr();
  const int* colind_remote = _mat_remote->innerIndexPtr();
  const T* values_remote = _mat_remote->valuePtr();
  const T* diagonal = _mat_diagonal->data();
  T* b_ptr = b.data();
  T* y_ptr = y.data();

#ifdef _OPENMP
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int row_offset = _row_split[tid];
    T* y_local = (tid == 0) ? y_ptr : _y_local[tid - 1];

    // Compute diagonal
    for (int i = _row_split[tid]; i < _row_split[tid + 1]; ++i)
      y_ptr[i] = diagonal[i] * b_ptr[i];
    #pragma omp barrier

    // Compute symmetric SpMV on local block - local vectors phase
    for (int i = _row_split[tid]; i < _row_split[tid + 1]; ++i)
    {
      T y_tmp = 0;

      for (int j = rowptr[i]; j < rowptr[i + 1]; ++j)
      {
        int col = colind[j];
        T val = values[j];
        y_tmp += val * b_ptr[col];
        if (col < row_offset)
        {
          y_local[col] += val * b_ptr[i];
        }
        else
        {
          y_ptr[col] += val * b_ptr[i];
        }
      }

      y_ptr[i] += y_tmp;
    }

    // Finalise ghost updates
    #pragma omp master
    _col_map->update_finalise(b_ptr);
    #pragma omp barrier

    // Compute vanilla SpMV on remote block
    for (int i = _row_split[tid]; i < _row_split[tid + 1]; ++i)
    {
      T y_tmp = 0;
      for (int j = rowptr_remote[i]; j < rowptr_remote[i + 1]; ++j)
      {
        y_tmp += values_remote[j] * b_ptr[colind_remote[j]];
      }
      y_ptr[i] += y_tmp;
    }
    #pragma omp barrier

    // Compute symmetric SpMV on local block - reduction of conflicts phase
    for (int i = _map_start[tid]; i < _map_end[tid]; ++i)
    {
      int vid = _cnfl_map->vid[i];
      int pos = _cnfl_map->pos[i];
      y_ptr[pos] += _y_local[vid - 1][pos];
      _y_local[vid - 1][pos] = 0.0;
    }
  }
#else
  // Compute symmetric SpMV on local block
  for (int i = 0; i < _mat_local->rows(); ++i)
  {
    T y_tmp = diagonal[i] * b_ptr[i];

    for (int j = rowptr[i]; j < rowptr[i + 1]; ++j)
    {
      int col = colind[j];
      T val = values[j];
      y_tmp += val * b_ptr[col];
      y_ptr[col] += val * b_ptr[i];
    }

    y_ptr[i] = y_tmp;
  }

  // Finalise ghost updates
  _col_map->update_finalise(b_ptr);

  // Compute vanilla SpMV on remote block
  for (int i = 0; i < _mat_remote->rows(); ++i)
  {
    T y_tmp = 0;

    for (int j = rowptr_remote[i]; j < rowptr_remote[i + 1]; ++j)
    {
      y_tmp += values_remote[j] * b_ptr[colind_remote[j]];
    }

    y_ptr[i] += y_tmp;
  }
#endif // _OPENMP

  return y;
}
//---------------------
#ifdef EIGEN_USE_MKL_ALL
template <>
void Matrix<double>::mkl_init()
{
  sparse_status_t status = mkl_sparse_d_create_csr(
      &_mat_local_mkl, SPARSE_INDEX_BASE_ZERO, _mat_local->rows(),
      _mat_local->cols(), const_cast<MKL_INT*>(_mat_local->outerIndexPtr()),
      const_cast<MKL_INT*>(_mat_local->outerIndexPtr()) + 1,
      const_cast<MKL_INT*>(_mat_local->innerIndexPtr()),
      const_cast<double*>(_mat_local->valuePtr()));
  assert(status == SPARSE_STATUS_SUCCESS);
  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not create MKL matrix");

  status = mkl_sparse_optimize(_mat_local_mkl);
  assert(status == SPARSE_STATUS_SUCCESS);
  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not optimize MKL matrix");

  _mat_local_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  _mat_local_desc.diag = SPARSE_DIAG_NON_UNIT;

  if (_col_map->overlapping() && _mat_remote->nonZeros() > 0)
  {
    status = mkl_sparse_d_create_csr(
        &_mat_remote_mkl, SPARSE_INDEX_BASE_ZERO, _mat_remote->rows(),
        _mat_remote->cols(), const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()),
        const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()) + 1,
        const_cast<MKL_INT*>(_mat_remote->innerIndexPtr()),
        const_cast<double*>(_mat_remote->valuePtr()));
    assert(status == SPARSE_STATUS_SUCCESS);
    if (status != SPARSE_STATUS_SUCCESS)
      throw std::runtime_error("Could not create MKL matrix");

    status = mkl_sparse_optimize(_mat_remote_mkl);
    assert(status == SPARSE_STATUS_SUCCESS);
    if (status != SPARSE_STATUS_SUCCESS)
      throw std::runtime_error("Could not optimize MKL matrix");

    _mat_remote_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
    _mat_remote_desc.diag = SPARSE_DIAG_NON_UNIT;
  }
}
//----------------------
template <>
void Matrix<std::complex<double>>::mkl_init()
{
  sparse_status_t status = mkl_sparse_z_create_csr(
      &_mat_local_mkl, SPARSE_INDEX_BASE_ZERO, _mat_local->rows(),
      _mat_local->cols(), const_cast<MKL_INT*>(_mat_local->outerIndexPtr()),
      const_cast<MKL_INT*>(_mat_local->outerIndexPtr()) + 1,
      const_cast<MKL_INT*>(_mat_local->innerIndexPtr()),
      (MKL_Complex16*)_mat_local->valuePtr());
  assert(status == SPARSE_STATUS_SUCCESS);
  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not create MKL matrix");

  status = mkl_sparse_optimize(_mat_local_mkl);
  assert(status == SPARSE_STATUS_SUCCESS);
  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not optimize MKL matrix");

  _mat_local_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  _mat_local_desc.diag = SPARSE_DIAG_NON_UNIT;

  if (_col_map->overlapping() && _mat_remote->nonZeros() > 0)
  {
    status = mkl_sparse_z_create_csr(
        &_mat_remote_mkl, SPARSE_INDEX_BASE_ZERO, _mat_remote->rows(),
        _mat_remote->cols(), const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()),
        const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()) + 1,
        const_cast<MKL_INT*>(_mat_remote->innerIndexPtr()),
        (MKL_Complex16*)_mat_remote->valuePtr());
    assert(status == SPARSE_STATUS_SUCCESS);
    if (status != SPARSE_STATUS_SUCCESS)
      throw std::runtime_error("Could not create MKL matrix");

    status = mkl_sparse_optimize(_mat_remote_mkl);
    assert(status == SPARSE_STATUS_SUCCESS);
    if (status != SPARSE_STATUS_SUCCESS)
      throw std::runtime_error("Could not optimize MKL matrix");

    _mat_remote_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
    _mat_remote_desc.diag = SPARSE_DIAG_NON_UNIT;
  }
}
//----------------------
template <>
void Matrix<float>::mkl_init()
{
  sparse_status_t status = mkl_sparse_s_create_csr(
      &_mat_local_mkl, SPARSE_INDEX_BASE_ZERO, _mat_local->rows(),
      _mat_local->cols(), const_cast<MKL_INT*>(_mat_local->outerIndexPtr()),
      const_cast<MKL_INT*>(_mat_local->outerIndexPtr()) + 1,
      const_cast<MKL_INT*>(_mat_local->innerIndexPtr()),
      const_cast<float*>(_mat_local->valuePtr()));
  assert(status == SPARSE_STATUS_SUCCESS);
  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not create MKL matrix");

  status = mkl_sparse_optimize(_mat_local_mkl);
  assert(status == SPARSE_STATUS_SUCCESS);
  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not optimize MKL matrix");

  _mat_local_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  _mat_local_desc.diag = SPARSE_DIAG_NON_UNIT;

  if (_col_map->overlapping() && _mat_remote->nonZeros() > 0)
  {
    status = mkl_sparse_s_create_csr(
        &_mat_remote_mkl, SPARSE_INDEX_BASE_ZERO, _mat_remote->rows(),
        _mat_remote->cols(), const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()),
        const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()) + 1,
        const_cast<MKL_INT*>(_mat_remote->innerIndexPtr()),
        const_cast<float*>(_mat_remote->valuePtr()));
    assert(status == SPARSE_STATUS_SUCCESS);
    if (status != SPARSE_STATUS_SUCCESS)
      throw std::runtime_error("Could not create MKL matrix");

    status = mkl_sparse_optimize(_mat_remote_mkl);
    assert(status == SPARSE_STATUS_SUCCESS);
    if (status != SPARSE_STATUS_SUCCESS)
      throw std::runtime_error("Could not optimize MKL matrix");

    _mat_remote_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
    _mat_remote_desc.diag = SPARSE_DIAG_NON_UNIT;
  }
}
//----------------------
template <>
void Matrix<std::complex<float>>::mkl_init()
{
  sparse_status_t status = mkl_sparse_c_create_csr(
      &_mat_local_mkl, SPARSE_INDEX_BASE_ZERO, _mat_local->rows(),
      _mat_local->cols(), const_cast<MKL_INT*>(_mat_local->outerIndexPtr()),
      const_cast<MKL_INT*>(_mat_local->outerIndexPtr()) + 1,
      const_cast<MKL_INT*>(_mat_local->innerIndexPtr()),
      (MKL_Complex8*)_mat_local->valuePtr());
  assert(status == SPARSE_STATUS_SUCCESS);
  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not create MKL matrix");

  status = mkl_sparse_optimize(_mat_local_mkl);
  assert(status == SPARSE_STATUS_SUCCESS);
  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not optimize MKL matrix");

  _mat_local_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  _mat_local_desc.diag = SPARSE_DIAG_NON_UNIT;

  if (_col_map->overlapping() && _mat_remote->nonZeros() > 0)
  {
    status = mkl_sparse_c_create_csr(
        &_mat_remote_mkl, SPARSE_INDEX_BASE_ZERO, _mat_remote->rows(),
        _mat_remote->cols(), const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()),
        const_cast<MKL_INT*>(_mat_remote->outerIndexPtr()) + 1,
        const_cast<MKL_INT*>(_mat_remote->innerIndexPtr()),
        (MKL_Complex8*)_mat_remote->valuePtr());
    assert(status == SPARSE_STATUS_SUCCESS);
    if (status != SPARSE_STATUS_SUCCESS)
      throw std::runtime_error("Could not create MKL matrix");

    status = mkl_sparse_optimize(_mat_remote_mkl);
    assert(status == SPARSE_STATUS_SUCCESS);
    if (status != SPARSE_STATUS_SUCCESS)
      throw std::runtime_error("Could not optimize MKL matrix");

    _mat_remote_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
    _mat_remote_desc.diag = SPARSE_DIAG_NON_UNIT;
  }
}
//----------------------
template <>
Eigen::VectorXd Matrix<double>::operator*(Eigen::VectorXd& b) const
{
  Eigen::VectorXd y(_mat_local->rows());
  if (_symmetric)
  {
    y = spmv_sym(b);
  }
  else if (_col_map->overlapping() && _mat_remote->nonZeros() > 0)
  {
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mat_local_mkl,
                    _mat_local_desc, b.data(), 0.0, y.data());
    _col_map->update_finalise(b.data());
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mat_remote_mkl,
                    _mat_remote_desc, b.data(), 1.0, y.data());
  }
  else
  {
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mat_local_mkl,
                    _mat_local_desc, b.data(), 0.0, y.data());
  }

  return y;
}
//---------------------
template <>
Eigen::VectorXd Matrix<double>::transpmult(const Eigen::VectorXd& b) const
{
  Eigen::VectorXd y(_mat_local->cols());
  if (_symmetric)
  {
    throw std::runtime_error(
        "transpmult() operation not yet implemented for symmetric matrices");
  }
  else
  {
    mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, _mat_local_mkl,
                    _mat_local_desc, b.data(), 0.0, y.data());
  }

  return y;
}
//----------------------
template <>
Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>
    Matrix<std::complex<double>>::
    operator*(Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>& b) const
{
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> y(_mat_local->rows());
  const MKL_Complex16 one({1.0, 0.0}), zero({0.0, 0.0});
  if (_symmetric)
  {
    throw std::runtime_error("Multiplication not yet implemented for symmetric "
                             "matrices for complex data");
  }
  else if (_col_map->overlapping() && _mat_remote->nonZeros() > 0)
  {
    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, one, _mat_local_mkl,
                    _mat_local_desc, (MKL_Complex16*)b.data(), zero,
                    (MKL_Complex16*)y.data());
    _col_map->update_finalise(b.data());
    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, one, _mat_remote_mkl,
                    _mat_remote_desc, (MKL_Complex16*)b.data(), one,
                    (MKL_Complex16*)y.data());
  }
  else
  {
    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, one, _mat_local_mkl,
                    _mat_local_desc, (MKL_Complex16*)b.data(), zero,
                    (MKL_Complex16*)y.data());
  }

  return y;
}
//----------------------
template <>
Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>
Matrix<std::complex<double>>::transpmult(
    const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>& b) const
{
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> y(_mat_local->rows());
  const MKL_Complex16 one({1.0, 0.0}), zero({0.0, 0.0});
  if (_symmetric)
  {
    throw std::runtime_error(
        "transpmult() operation not yet implemented for symmetric matrices");
  }
  else
  {
    mkl_sparse_z_mv(SPARSE_OPERATION_TRANSPOSE, one, _mat_local_mkl,
                    _mat_local_desc, (MKL_Complex16*)b.data(), zero,
                    (MKL_Complex16*)y.data());
  }
  return y;
}
//----------------------
template <>
Eigen::VectorXf Matrix<float>::operator*(Eigen::VectorXf& b) const
{
  Eigen::VectorXf y(_mat_local->rows());
  if (_symmetric)
  {
    y = spmv_sym(b);
  }
  else if (_col_map->overlapping() && _mat_remote->nonZeros() > 0)
  {
    mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mat_local_mkl,
                    _mat_local_desc, b.data(), 0.0, y.data());
    _col_map->update_finalise(b.data());
    mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mat_remote_mkl,
                    _mat_remote_desc, b.data(), 1.0, y.data());
  }
  else
  {
    mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, _mat_local_mkl,
                    _mat_local_desc, b.data(), 0.0, y.data());
  }

  return y;
}
//---------------------
template <>
Eigen::VectorXf Matrix<float>::transpmult(const Eigen::VectorXf& b) const
{
  Eigen::VectorXf y(_mat_local->cols());
  if (_symmetric)
  {
    throw std::runtime_error(
        "transpmult() operation not yet implemented for symmetric matrices");
  }
  else
  {
    mkl_sparse_s_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, _mat_local_mkl,
                    _mat_local_desc, b.data(), 0.0, y.data());
  }

  return y;
}
//----------------------
template <>
Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1>
    Matrix<std::complex<float>>::
    operator*(Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1>& b) const
{
  Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1> y(_mat_local->rows());
  const MKL_Complex8 one({1.0, 0.0}), zero({0.0, 0.0});
  if (_symmetric)
  {
    throw std::runtime_error("Multiplication not yet implemented for symmetric "
                             "matrices for complex data");
  }
  else if (_col_map->overlapping() && _mat_remote->nonZeros() > 0)
  {
    mkl_sparse_c_mv(SPARSE_OPERATION_NON_TRANSPOSE, one, _mat_local_mkl,
                    _mat_local_desc, (MKL_Complex8*)b.data(), zero,
                    (MKL_Complex8*)y.data());
    _col_map->update_finalise(b.data());
    mkl_sparse_c_mv(SPARSE_OPERATION_NON_TRANSPOSE, one, _mat_remote_mkl,
                    _mat_remote_desc, (MKL_Complex8*)b.data(), one,
                    (MKL_Complex8*)y.data());
  }
  else
  {
    mkl_sparse_c_mv(SPARSE_OPERATION_NON_TRANSPOSE, one, _mat_local_mkl,
                    _mat_local_desc, (MKL_Complex8*)b.data(), zero,
                    (MKL_Complex8*)y.data());
  }

  return y;
}
//----------------------
template <>
Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1>
Matrix<std::complex<float>>::transpmult(
    const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1>& b) const
{
  Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1> y(_mat_local->rows());
  const MKL_Complex8 one({1.0, 0.0}), zero({0.0, 0.0});
  if (_symmetric)
  {
    throw std::runtime_error(
        "transpmult() operation not yet implemented for symmetric matrices");
  }
  else
  {
    mkl_sparse_c_mv(SPARSE_OPERATION_TRANSPOSE, one, _mat_local_mkl,
                    _mat_local_desc, (MKL_Complex8*)b.data(), zero,
                    (MKL_Complex8*)y.data());
  }
  return y;
}
#endif // _EIGEN_USE_MKL_ALL

//-----------------------------------------------------------------------------
// Explicit instantiation
template class spmv::Matrix<float>;
template class spmv::Matrix<double>;
template class spmv::Matrix<std::complex<float>>;
template class spmv::Matrix<std::complex<double>>;

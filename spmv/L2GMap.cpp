// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "L2GMap.h"
#include "mpi_type.h"
#include <algorithm>
#include <complex>
#include <iostream>
#include <set>
#include <vector>

using namespace spmv;

//-----------------------------------------------------------------------------
L2GMap::L2GMap(MPI_Comm comm, std::int64_t local_size,
               const std::vector<std::int64_t>& ghosts, bool p2p, bool overlap)
    : _ghosts(ghosts), _comm(comm), _p2p(p2p), _overlap(overlap), _req(nullptr),
      _send_buf(nullptr), _recv_buf(nullptr)
{
  int mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &_mpi_rank);

  _ranges.resize(mpi_size + 1);
  _ranges[0] = 0;
  MPI_Allgather(&local_size, 1, MPI_INT64_T, _ranges.data() + 1, 1, MPI_INT64_T,
                comm);
  for (int i = 0; i < mpi_size; ++i)
    _ranges[i + 1] += _ranges[i];

  const std::int64_t r0 = _ranges[_mpi_rank];
  const std::int64_t r1 = _ranges[_mpi_rank + 1];

  // Make sure ghosts are in order
  if (!std::is_sorted(_ghosts.begin(), _ghosts.end()))
    throw std::runtime_error("Ghosts must be sorted");

  // Get count on each process and local index
  std::vector<std::int32_t> ghost_count(mpi_size, 0);
  std::vector<std::int32_t> ghost_local;
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
  {
    const std::int64_t idx = _ghosts[i];

    if (idx >= r0 and idx < r1)
      throw std::runtime_error("Ghost index in local range");
    _global_to_local.insert({idx, local_size + i});

    auto it = std::upper_bound(_ranges.begin(), _ranges.end(), idx);
    assert(it != _ranges.end());
    const int p = it - _ranges.begin() - 1;
    ++ghost_count[p];
    assert(_ghosts[i] >= _ranges[p] and _ghosts[i] < _ranges[p + 1]);
    ghost_local.push_back(_ghosts[i] - _ranges[p]);
  }
  assert(ghost_local.size() == _ghosts.size());

  // Symmetrise neighbours. This ensures that both forward and reverse updates
  // will work.
  std::vector<std::int32_t> remote_count(mpi_size);
  MPI_Alltoall(ghost_count.data(), 1, MPI_INT, remote_count.data(), 1, MPI_INT,
               comm);

  for (std::size_t i = 0; i < ghost_count.size(); ++i)
  {
    const std::int32_t c = ghost_count[i];
    const std::int32_t rc = remote_count[i];
    if (c > 0)
    {
      _neighbours.push_back(i);
      _send_count.push_back(c);
    }
    else if (rc > 0)
    {
      _neighbours.push_back(i);
      _send_count.push_back(0);
    }
  }

  const int neighbour_size = _neighbours.size();
  MPI_Dist_graph_create_adjacent(comm, neighbour_size, _neighbours.data(),
                                 MPI_UNWEIGHTED, neighbour_size,
                                 _neighbours.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &_neighbour_comm);

  _recv_count.resize(neighbour_size);
  if (neighbour_size == 0)
  {
    // Needed for OpenMPI
    _send_count = {0};
    _recv_count = {0};
  }

  // Send NNZs by Alltoall - these will be the receive counts for incoming
  // index/values
  MPI_Neighbor_alltoall(_send_count.data(), 1, MPI_INT, _recv_count.data(), 1,
                        MPI_INT, _neighbour_comm);

  _send_offset = {0};
  for (int c : _send_count)
    _send_offset.push_back(_send_offset.back() + c);

  _recv_offset = {0};
  for (int c : _recv_count)
    _recv_offset.push_back(_recv_offset.back() + c);
  int count = _recv_offset.back();

  _indexbuf.resize(count);

  // Send global indices to remote processes that own them
  int err = MPI_Neighbor_alltoallv(
      ghost_local.data(), _send_count.data(), _send_offset.data(), MPI_INT32_T,
      _indexbuf.data(), _recv_count.data(), _recv_offset.data(), MPI_INT32_T,
      _neighbour_comm);
  if (err != MPI_SUCCESS)
    throw std::runtime_error("MPI failure");

  // Add local_range onto _send_offset (ghosts will be at end of range) in case
  // of blocking communication
  if (!overlap)
    for (std::int32_t& s : _send_offset)
      s += local_size;

  if (_p2p)
  {
    MPI_Comm_free(&_neighbour_comm);
    if (_overlap)
      _req = new MPI_Request[2 * neighbour_size];
    else
      _req = new MPI_Request[neighbour_size];
  }
  else
  {
    if (_overlap)
      _req = new MPI_Request;
  }
}
//-----------------------------------------------------------------------------
L2GMap::~L2GMap()
{
  if (!_p2p)
    MPI_Comm_free(&_neighbour_comm);
  if (_p2p)
    delete[] _req;
  else if (_overlap)
    delete _req;
  if (_overlap)
  {
    operator delete(_send_buf);
    operator delete(_recv_buf);
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_collective(T* vec_data) const
{
  assert(_overlap == false && _p2p == false);
  const int num_indices = _indexbuf.size();

  // Get data from local indices to send to other processes, landing in their
  // ghost region
  std::vector<T> databuf(num_indices);
  std::transform(std::begin(_indexbuf), std::end(_indexbuf),
                 std::begin(databuf),
                 [vec_data](auto i) { return vec_data[i]; });

  // Send actual values - NB meaning of _send and _recv count/offset is
  // reversed
  MPI_Datatype data_type = mpi_type<T>();
  int err = MPI_Neighbor_alltoallv(databuf.data(), _recv_count.data(),
                                   _recv_offset.data(), data_type, vec_data,
                                   _send_count.data(), _send_offset.data(),
                                   data_type, _neighbour_comm);
  if (err != MPI_SUCCESS)
    throw std::runtime_error("MPI failure");
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_collective_start(T* vec_data) const
{
  assert(_overlap == true && _p2p == false);
  const int num_indices = _indexbuf.size();

  // Allocate send and receive buffers, if this is the first call to update
  if (_send_buf == nullptr)
    _send_buf = ::operator new(num_indices * sizeof(T));
  if (_recv_buf == nullptr)
    _recv_buf = ::operator new(num_ghosts() * sizeof(T));

  // Get data from local indices to send to other processes, landing in their
  // ghost region
  T* ptr = static_cast<T*>(_send_buf);
  for (int i = 0; i < num_indices; ++i)
    ptr[i] = vec_data[_indexbuf[i]];

  // Send actual values - NB meaning of _send and _recv count/offset is
  // reversed
  MPI_Datatype data_type = mpi_type<T>();
  int err = MPI_Ineighbor_alltoallv(_send_buf, _recv_count.data(),
                                    _recv_offset.data(), data_type, _recv_buf,
                                    _send_count.data(), _send_offset.data(),
                                    data_type, _neighbour_comm, _req);
  if (err != MPI_SUCCESS)
    throw std::runtime_error("MPI failure");
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_collective_end(T* vec_data) const
{
  assert(_overlap == true && _p2p == false);
  MPI_Wait(_req, MPI_STATUS_IGNORE);
  // Copy ghosts from intermediate buffer to vector
  memcpy((void*)(vec_data + local_size()), _recv_buf, num_ghosts() * sizeof(T));
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_p2p(T* vec_data) const
{
  assert(_overlap == false && _p2p == true);
  const int num_indices = _indexbuf.size();

  // Get data from local indices to send to other processes, landing in their
  // ghost region
  std::vector<T> databuf(num_indices);
  std::transform(std::begin(_indexbuf), std::end(_indexbuf),
                 std::begin(databuf),
                 [vec_data](auto i) { return vec_data[i]; });

  // Send actual values - NB meaning of _send and _recv count/offset is
  // reversed
  MPI_Datatype data_type = mpi_type<T>();
  const int num_neighbours = _neighbours.size();
  for (int i = 0; i < num_neighbours; ++i)
  {
    T* recv_buf = vec_data + _send_offset[i];
    MPI_Irecv(recv_buf, _send_count[i], data_type, _neighbours[i], 0, _comm,
              &(_req[i]));
  }

  for (int i = 0; i < num_neighbours; ++i)
  {
    T* send_buf = databuf.data() + _recv_offset[i];
    MPI_Send(send_buf, _recv_count[i], data_type, _neighbours[i], 0, _comm);
  }

  MPI_Waitall(num_neighbours, _req, MPI_STATUSES_IGNORE);
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_p2p_start(T* vec_data) const
{
  assert(_overlap == true && _p2p == true);
  const int num_indices = _indexbuf.size();

  // Allocate send and receive buffers, if this is the first call to update
  if (_send_buf == nullptr)
    _send_buf = ::operator new(num_indices * sizeof(T));
  if (_recv_buf == nullptr)
    _recv_buf = ::operator new(num_ghosts() * sizeof(T));

  // Get data from local indices to send to other processes, landing in their
  // ghost region
  T* ptr = static_cast<T*>(_send_buf);
  for (int i = 0; i < num_indices; ++i)
    ptr[i] = vec_data[_indexbuf[i]];

  // Send actual values - NB meaning of _send and _recv count/offset is
  // reversed
  MPI_Datatype data_type = mpi_type<T>();
  const int num_neighbours = _neighbours.size();
  for (int i = 0; i < num_neighbours; ++i)
  {
    T* recv_buf = static_cast<T*>(_recv_buf) + _send_offset[i];
    MPI_Irecv(recv_buf, _send_count[i], data_type, _neighbours[i], 0, _comm,
              &(_req[i]));
  }

  for (int i = 0; i < num_neighbours; ++i)
  {
    T* send_buf = static_cast<T*>(_send_buf) + _recv_offset[i];
    MPI_Isend(send_buf, _recv_count[i], data_type, _neighbours[i], 0, _comm,
              &(_req[num_neighbours + i]));
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_p2p_end(T* vec_data) const
{
  assert(_overlap == true && _p2p == true);
  MPI_Waitall(2 * _neighbours.size(), _req, MPI_STATUSES_IGNORE);
  // Copy ghosts from intermediate buffer to vector
  memcpy((void*)(vec_data + local_size()), _recv_buf, num_ghosts() * sizeof(T));
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update(T* vec_data) const
{
  if (_overlap)
  {
    if (_p2p)
      update_p2p_start(vec_data);
    else
      update_collective_start(vec_data);
  }
  else
  {
    if (_p2p)
      update_p2p(vec_data);
    else
      update_collective(vec_data);
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_finalise(T* vec_data) const
{
  if (!_overlap)
    return;
  if (_p2p)
    update_p2p_end(vec_data);
  else
    update_collective_end(vec_data);
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::reverse_update_collective(T* vec_data) const
{
  const int num_indices = _indexbuf.size();

  // Send values from ghost region of vector to remotes
  // accumulating in local vector.
  std::vector<T> databuf(num_indices);
  MPI_Datatype data_type = mpi_type<T>();
  int err = MPI_Neighbor_alltoallv(
      vec_data, _send_count.data(), _send_offset.data(), data_type,
      databuf.data(), _recv_count.data(), _recv_offset.data(), data_type,
      _neighbour_comm);
  if (err != MPI_SUCCESS)
    throw std::runtime_error("MPI failure");

  for (int i = 0; i < num_indices; ++i)
    vec_data[_indexbuf[i]] += databuf[i];
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::reverse_update_p2p(T* vec_data) const
{
  const int num_indices = _indexbuf.size();

  // Send values from ghost region of vector to remotes
  // accumulating in local vector.
  std::vector<T> databuf(num_indices);
  MPI_Datatype data_type = mpi_type<T>();
  const int num_neighbours = _neighbours.size();
  MPI_Request* rq = _req;

  for (int i = 0; i < num_neighbours; ++i)
  {
    T* recv_buf = databuf.data() + _recv_offset[i];
    MPI_Irecv(recv_buf, _recv_count[i], data_type, _neighbours[i], 0, _comm,
              rq++);
  }

  for (int i = 0; i < num_neighbours; ++i)
  {
    T* send_buf = vec_data + _send_offset[i];
    MPI_Isend(send_buf, _send_count[i], data_type, _neighbours[i], 0, _comm,
              rq++);
  }

  MPI_Waitall(num_neighbours, _req, MPI_STATUS_IGNORE);

  for (int i = 0; i < num_indices; ++i)
    vec_data[_indexbuf[i]] += databuf[i];
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::reverse_update(T* vec_data) const
{
  if (_p2p)
    reverse_update_p2p(vec_data);
  else
    reverse_update_collective(vec_data);
}
//-----------------------------------------------------------------------------
std::int32_t L2GMap::global_to_local(std::int64_t i) const
{
  const std::int64_t r0 = _ranges[_mpi_rank];
  const std::int64_t r1 = _ranges[_mpi_rank + 1];

  if (i >= r0 and i < r1)
    return (i - r0);
  else
  {
    auto it = _global_to_local.find(i);
    assert(it != _global_to_local.end());
    return it->second;
  }
}
//-----------------------------------------------------------------------------
std::int32_t L2GMap::local_size() const
{
  return (_ranges[_mpi_rank + 1] - _ranges[_mpi_rank]);
}
//-----------------------------------------------------------------------------
std::int32_t L2GMap::num_ghosts() const { return _ghosts.size(); }
//-----------------------------------------------------------------------------
std::int64_t L2GMap::global_size() const { return _ranges.back(); }
//-----------------------------------------------------------------------------
std::int64_t L2GMap::global_offset() const { return _ranges[_mpi_rank]; }
//-----------------------------------------------------------------------------
// Explicit instantiation
template void spmv::L2GMap::update<double>(double*) const;
template void spmv::L2GMap::update<float>(float*) const;
template void
spmv::L2GMap::update<std::complex<float>>(std::complex<float>*) const;
template void
spmv::L2GMap::update<std::complex<double>>(std::complex<double>*) const;
template void spmv::L2GMap::update_finalise<double>(double*) const;
template void spmv::L2GMap::update_finalise<float>(float*) const;
template void
spmv::L2GMap::update_finalise<std::complex<float>>(std::complex<float>*) const;
template void spmv::L2GMap::update_finalise<std::complex<double>>(
    std::complex<double>*) const;
template void spmv::L2GMap::reverse_update<double>(double*) const;
template void spmv::L2GMap::reverse_update<float>(float*) const;
template void
spmv::L2GMap::reverse_update<std::complex<float>>(std::complex<float>*) const;
template void
spmv::L2GMap::reverse_update<std::complex<double>>(std::complex<double>*) const;

// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <set>
#include <vector>

#include "L2GMap.h"
#include "device_executor.h"

using namespace spmv;
//-----------------------------------------------------------------------------
L2GMap::L2GMap(MPI_Comm comm, std::int64_t local_size,
               const std::vector<std::int64_t>& ghosts,
               std::shared_ptr<DeviceExecutor> exec, CommunicationModel cm)
    : _exec(exec), _ghosts(ghosts), _comm(comm), _cm(cm)
{
  if (cm == CommunicationModel::shmem_nodup) {
    int mpi_size;
    CHECK_MPI(MPI_Comm_size(comm, &mpi_size));
    CHECK_MPI(MPI_Comm_rank(comm, &_mpi_rank));

    // Create node-level communicators
    CHECK_MPI(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                  &_node_comm));

    int mpi_node_size;
    CHECK_MPI(MPI_Comm_size(_node_comm, &mpi_node_size));
    CHECK_MPI(MPI_Comm_rank(_node_comm, &_mpi_node_rank));

    // Create MPI groups for global and node communicator
    MPI_Group global_group, node_group;
    CHECK_MPI(MPI_Comm_group(comm, &global_group));
    CHECK_MPI(MPI_Comm_group(_node_comm, &node_group));

    // Gather row/col ranges of other processes
    _ranges.resize(mpi_size + 1);
    _ranges[0] = 0;
    CHECK_MPI(MPI_Allgather(&local_size, 1, MPI_INT64_T, _ranges.data() + 1, 1,
                            MPI_INT64_T, comm));
    for (int i = 0; i < mpi_size; ++i)
      _ranges[i + 1] += _ranges[i];

    const std::int64_t r0 = _ranges[_mpi_rank];
    const std::int64_t r1 = _ranges[_mpi_rank + 1];

    // Make sure ghosts are sorted
    if (!std::is_sorted(_ghosts.begin(), _ghosts.end()))
      throw std::runtime_error("Ghosts must be sorted");

    // Group ghosts depending on whether the owner process is on-node or
    // off-node
    std::vector<std::int32_t> ghost_count(mpi_size, 0);
    _send_count_on_node.resize(mpi_node_size, 0);
    std::vector<std::int32_t> ghost_count_off_node(mpi_size, 0);
    //    std::vector<std::int32_t> ghost_local_indices_on_node;
    std::vector<std::int32_t> ghost_local_indices_off_node;
    for (std::size_t i = 0; i < _ghosts.size(); ++i) {
      const std::int64_t idx = _ghosts[i];

      if (idx >= r0 and idx < r1)
        throw std::runtime_error("Ghost index in local range");
      _global_to_local.insert({idx, local_size + i});

      auto it = std::upper_bound(_ranges.begin(), _ranges.end(), idx);
      assert(it != _ranges.end());
      const int rank = it - _ranges.begin() - 1;
      int node_rank;
      CHECK_MPI(MPI_Group_translate_ranks(global_group, 1, &rank, node_group,
                                          &node_rank));
      // If translated rank is MPI_UNDEFINED, then this is a off-node process
      if (node_rank == MPI_UNDEFINED) {
        ++ghost_count_off_node[rank];
        ghost_local_indices_off_node.push_back(_ghosts[i] - _ranges[rank]);
      } else {
        ++_send_count_on_node[node_rank];
        ghost_local_indices_on_node.push_back(_ghosts[i] - _ranges[rank]);
      }
      ++ghost_count[rank];
      assert(_ghosts[i] >= _ranges[rank] && _ghosts[i] < _ranges[rank + 1]);
    }
    assert(ghost_local_indices_on_node.size()
               + ghost_local_indices_off_node.size()
           == _ghosts.size());

    /////////////////////////////////////////////////////////////////////////////
    // Setup communcation with off-node neighbours
    // Symmetrise off-node neighbours. This ensures that both forward and
    // reverse updates will work. Symmetrise neighbours. This ensures that both
    // forward and reverse updates will work.
    std::vector<std::int32_t> remote_count_off_node(mpi_size, 0);
    CHECK_MPI(MPI_Alltoall(ghost_count_off_node.data(), 1, MPI_INT,
                           remote_count_off_node.data(), 1, MPI_INT, comm));

    std::vector<int> neighbours_off_node;
    for (std::size_t i = 0; i < ghost_count_off_node.size(); ++i) {
      const std::int32_t c = ghost_count_off_node[i];
      const std::int32_t rc = remote_count_off_node[i];
      if (c > 0 || rc > 0)
        neighbours_off_node.push_back(i);
      if (c > 0) {
        _send_count.push_back(c);
        if (rc == 0)
          _recv_count.push_back(0);
      }
      if (rc > 0) {
        _recv_count.push_back(rc);
        if (c == 0)
          _send_count.push_back(0);
      }
    }

    // Create neighbourhood graph for off-node communication
    CHECK_MPI(MPI_Dist_graph_create_adjacent(
        comm, neighbours_off_node.size(), neighbours_off_node.data(),
        MPI_UNWEIGHTED, neighbours_off_node.size(), neighbours_off_node.data(),
        MPI_UNWEIGHTED, MPI_INFO_NULL, false, &_neighbour_comm));

    if (neighbours_off_node.size() == 0) {
      // Needed for OpenMPI
      _send_count = {0};
      _recv_count = {0};
    }

    _send_offset = {0};
    for (int c : _send_count)
      _send_offset.push_back(_send_offset.back() + c);

    _recv_offset = {0};
    for (int c : _recv_count)
      _recv_offset.push_back(_recv_offset.back() + c);

    std::vector<std::int32_t> indexbuf(_recv_offset.back());

    // Send global indices to remote processes that own them
    int err = MPI_Neighbor_alltoallv(
        ghost_local_indices_off_node.data(), _send_count.data(),
        _send_offset.data(), MPI_INT32_T, indexbuf.data(), _recv_count.data(),
        _recv_offset.data(), MPI_INT32_T, _neighbour_comm);
    if (err != MPI_SUCCESS)
      throw std::runtime_error("MPI failure");

    // Add local_range onto _send_offset (ghosts will be at end of range)
    for (std::int32_t& s : _send_offset)
      s += local_size;

    //////////////////////////////////////////////////////////////////////////
    // Setup communcation with on-node neighbours
    for (int i = 0; i < mpi_node_size; ++i) {
      const std::int32_t c = _send_count_on_node[i];
      if (c > 0) {
        _senders_on_node.push_back(i);
      }
    }

    _send_offset_on_node = {0};
    for (int c : _send_count_on_node)
      _send_offset_on_node.push_back(_send_offset_on_node.back() + c);

    // Add local_range onto _send_offset (ghosts will be at end of range)
    for (std::int32_t& s : _send_offset_on_node)
      s += local_size;
  } else if (cm == CommunicationModel::shmem) {
    int mpi_size;
    CHECK_MPI(MPI_Comm_size(comm, &mpi_size));
    CHECK_MPI(MPI_Comm_rank(comm, &_mpi_rank));

    // Create node-level communicators
    CHECK_MPI(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                  &_node_comm));

    int mpi_node_size;
    CHECK_MPI(MPI_Comm_size(_node_comm, &mpi_node_size));
    CHECK_MPI(MPI_Comm_rank(_node_comm, &_mpi_node_rank));

    // Create MPI groups for global and node communicator
    MPI_Group global_group, node_group;
    CHECK_MPI(MPI_Comm_group(comm, &global_group));
    CHECK_MPI(MPI_Comm_group(_node_comm, &node_group));

    // Gather row/col ranges of other processes
    _ranges.resize(mpi_size + 1);
    _ranges[0] = 0;
    CHECK_MPI(MPI_Allgather(&local_size, 1, MPI_INT64_T, _ranges.data() + 1, 1,
                            MPI_INT64_T, comm));
    for (int i = 0; i < mpi_size; ++i)
      _ranges[i + 1] += _ranges[i];

    const std::int64_t r0 = _ranges[_mpi_rank];
    const std::int64_t r1 = _ranges[_mpi_rank + 1];

    // Make sure ghosts are sorted
    if (!std::is_sorted(_ghosts.begin(), _ghosts.end()))
      throw std::runtime_error("Ghosts must be sorted");

    // Group ghosts depending on whether the owner process is on-node or
    // off-node
    std::vector<std::int32_t> ghost_count(mpi_size, 0);
    _send_count_on_node.resize(mpi_node_size, 0);
    std::vector<std::int32_t> ghost_count_off_node(mpi_size, 0);
    //    std::vector<std::int32_t> ghost_local_indices_on_node;
    std::vector<std::int32_t> ghost_local_indices_off_node;
    for (std::size_t i = 0; i < _ghosts.size(); ++i) {
      const std::int64_t idx = _ghosts[i];

      if (idx >= r0 and idx < r1)
        throw std::runtime_error("Ghost index in local range");
      _global_to_local.insert({idx, local_size + i});

      auto it = std::upper_bound(_ranges.begin(), _ranges.end(), idx);
      assert(it != _ranges.end());
      const int rank = it - _ranges.begin() - 1;
      int node_rank;
      CHECK_MPI(MPI_Group_translate_ranks(global_group, 1, &rank, node_group,
                                          &node_rank));
      // If translated rank is MPI_UNDEFINED, then this is a off-node process
      if (node_rank == MPI_UNDEFINED) {
        ++ghost_count_off_node[rank];
        ghost_local_indices_off_node.push_back(_ghosts[i] - _ranges[rank]);
      } else {
        ++_send_count_on_node[node_rank];
        ghost_local_indices_on_node.push_back(_ghosts[i] - _ranges[rank]);
      }
      ++ghost_count[rank];
      assert(_ghosts[i] >= _ranges[rank] && _ghosts[i] < _ranges[rank + 1]);
    }
    assert(ghost_local_indices_on_node.size()
               + ghost_local_indices_off_node.size()
           == _ghosts.size());

    /////////////////////////////////////////////////////////////////////////////
    // Setup communcation with off-node neighbours
    // Symmetrise off-node neighbours. This ensures that both forward and
    // reverse updates will work. Symmetrise neighbours. This ensures that both
    // forward and reverse updates will work.
    std::vector<std::int32_t> remote_count_off_node(mpi_size, 0);
    CHECK_MPI(MPI_Alltoall(ghost_count_off_node.data(), 1, MPI_INT,
                           remote_count_off_node.data(), 1, MPI_INT, comm));

    std::vector<int> neighbours_off_node;
    for (std::size_t i = 0; i < ghost_count_off_node.size(); ++i) {
      const std::int32_t c = ghost_count_off_node[i];
      const std::int32_t rc = remote_count_off_node[i];
      if (c > 0 || rc > 0)
        neighbours_off_node.push_back(i);
      if (c > 0) {
        _send_count.push_back(c);
        if (rc == 0)
          _recv_count.push_back(0);
      }
      if (rc > 0) {
        _recv_count.push_back(rc);
        if (c == 0)
          _send_count.push_back(0);
      }
    }

    // Create neighbourhood graph for off-node communication
    CHECK_MPI(MPI_Dist_graph_create_adjacent(
        comm, neighbours_off_node.size(), neighbours_off_node.data(),
        MPI_UNWEIGHTED, neighbours_off_node.size(), neighbours_off_node.data(),
        MPI_UNWEIGHTED, MPI_INFO_NULL, false, &_neighbour_comm));

    if (neighbours_off_node.size() == 0) {
      // Needed for OpenMPI
      _send_count = {0};
      _recv_count = {0};
    }

    _send_offset = {0};
    for (int c : _send_count)
      _send_offset.push_back(_send_offset.back() + c);

    _recv_offset = {0};
    for (int c : _recv_count)
      _recv_offset.push_back(_recv_offset.back() + c);

    std::vector<std::int32_t> indexbuf(_recv_offset.back());

    // Send global indices to remote processes that own them
    CHECK_MPI(MPI_Neighbor_alltoallv(
        ghost_local_indices_off_node.data(), _send_count.data(),
        _send_offset.data(), MPI_INT32_T, indexbuf.data(), _recv_count.data(),
        _recv_offset.data(), MPI_INT32_T, _neighbour_comm));

    // Add local_range onto _send_offset (ghosts will be at end of range)
    for (std::int32_t& s : _send_offset)
      s += local_size;

    //////////////////////////////////////////////////////////////////////////
    // Setup communcation with on-node neighbours
    _recv_count_on_node.resize(mpi_node_size, 0);
    CHECK_MPI(MPI_Alltoall(_send_count_on_node.data(), 1, MPI_INT,
                           _recv_count_on_node.data(), 1, MPI_INT, _node_comm));

    // _senders_on_node are the processes that I need data from
    // _receivers_on_node are the processes that need data from me
    for (int i = 0; i < mpi_node_size; ++i) {
      const std::int32_t c = _send_count_on_node[i];
      const std::int32_t rc = _recv_count_on_node[i];
      if (c > 0) {
        _senders_on_node.push_back(i);
      }
      if (rc > 0) {
        _receivers_on_node.push_back(i);
      }
    }

    _send_offset_on_node = {0};
    for (int c : _send_count_on_node)
      _send_offset_on_node.push_back(_send_offset_on_node.back() + c);

    _recv_offset_on_node = {0};
    for (int c : _recv_count_on_node)
      _recv_offset_on_node.push_back(_recv_offset_on_node.back() + c);
    _indexbuf_on_node.resize(_recv_offset_on_node.back());

    if (mpi_node_size == 0) {
      // Needed for OpenMPI
      _recv_count_on_node = {0};
      _send_count_on_node = {0};
    }

    // Send global indices to local processes that own them
    CHECK_MPI(MPI_Alltoallv(
        ghost_local_indices_on_node.data(), _send_count_on_node.data(),
        _send_offset_on_node.data(), MPI_INT32_T, _indexbuf_on_node.data(),
        _recv_count_on_node.data(), _recv_offset_on_node.data(), MPI_INT32_T,
        _node_comm));

    // FIXME Gather approach
    // I need to know where in each neighbor's window I need to read data from
    _sender_window_offset.resize(mpi_node_size, 0);
    CHECK_MPI(MPI_Alltoall(_send_offset_on_node.data(), 1, MPI_INT,
                           _sender_window_offset.data(), 1, MPI_INT,
                           _node_comm));

    // Add local_range onto _send_offset (ghosts will be at end of range)
    for (std::int32_t& s : _send_offset_on_node)
      s += local_size;
  } else {
    int mpi_size;
    CHECK_MPI(MPI_Comm_size(comm, &mpi_size));
    CHECK_MPI(MPI_Comm_rank(comm, &_mpi_rank));

    _ranges.resize(mpi_size + 1);
    _ranges[0] = 0;
    CHECK_MPI(MPI_Allgather(&local_size, 1, MPI_INT64_T, _ranges.data() + 1, 1,
                            MPI_INT64_T, comm));
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
    for (std::size_t i = 0; i < _ghosts.size(); ++i) {
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
    CHECK_MPI(MPI_Alltoall(ghost_count.data(), 1, MPI_INT, remote_count.data(),
                           1, MPI_INT, comm));

    for (int i = 0; i < mpi_size; ++i) {
      const std::int32_t c = ghost_count[i];
      const std::int32_t rc = remote_count[i];
      // if (c > 0) {
      // 	_neighbours.push_back(i);
      // 	_send_count.push_back(c);
      // } else if (rc > 0) {
      // 	_neighbours.push_back(i);
      // 	_send_count.push_back(0);
      // }
      if (c > 0 || rc > 0)
        _neighbours.push_back(i);
      if (c > 0) {
        _send_count.push_back(c);
        if (rc == 0)
          _recv_count.push_back(0);
      }
      if (rc > 0) {
        _recv_count.push_back(rc);
        if (c == 0)
          _send_count.push_back(0);
      }
    }

    const int neighbour_size = _neighbours.size();
    CHECK_MPI(MPI_Dist_graph_create_adjacent(
        comm, neighbour_size, _neighbours.data(), MPI_UNWEIGHTED,
        neighbour_size, _neighbours.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
        false, &_neighbour_comm));

    _recv_count.resize(neighbour_size);
    if (neighbour_size == 0) {
      // Needed for OpenMPI
      _send_count = {0};
      _recv_count = {0};
    }

    // Send NNZs by Alltoall - these will be the receive counts for incoming
    // index/values
    // MPI_Neighbor_alltoall(_send_count.data(), 1, MPI_INT, _recv_count.data(),
    // 1, MPI_INT, _neighbour_comm);

    _send_offset = {0};
    for (int c : _send_count)
      _send_offset.push_back(_send_offset.back() + c);

    _recv_offset = {0};
    for (int c : _recv_count)
      _recv_offset.push_back(_recv_offset.back() + c);
    int count = _recv_offset.back();

    std::vector<std::int32_t> indexbuf(count);

    // Send global indices to remote processes that own them
    CHECK_MPI(MPI_Neighbor_alltoallv(
        ghost_local.data(), _send_count.data(), _send_offset.data(),
        MPI_INT32_T, indexbuf.data(), _recv_count.data(), _recv_offset.data(),
        MPI_INT32_T, _neighbour_comm));

    // Determine offsets in window of each neighbour for one-sided communication
    if (_cm == CommunicationModel::onesided_put_active
        || _cm == CommunicationModel::onesided_put_passive) {
      _recv_win_offset.resize(neighbour_size);
      CHECK_MPI(MPI_Neighbor_alltoall(_send_offset.data(), 1, MPI_INT,
                                      _recv_win_offset.data(), 1, MPI_INT,
                                      _neighbour_comm));
    }

    // Add local_range onto _send_offset (ghosts will be at end of range) in
    // case of blocking communication
    for (std::int32_t& s : _send_offset)
      s += local_size;

    if (!(_cm == CommunicationModel::collective_blocking
          || _cm == CommunicationModel::collective_nonblocking))
      CHECK_MPI(MPI_Comm_free(&_neighbour_comm));
    if (_cm == CommunicationModel::p2p_nonblocking)
      _req = new MPI_Request[2 * neighbour_size];
    if (_cm == CommunicationModel::p2p_blocking)
      _req = new MPI_Request[neighbour_size];
    if (_cm == CommunicationModel::collective_nonblocking)
      _req = new MPI_Request;

    _num_indices = indexbuf.size();
    if (_num_indices > 0) {
      _indexbuf = _exec->alloc<int>(indexbuf.size());
      _exec->copy_from<int>(_indexbuf, _exec->get_host(), indexbuf.data(),
                            indexbuf.size());
    }
  }
}
//-----------------------------------------------------------------------------
L2GMap::~L2GMap()
{
  if (_cm == CommunicationModel::collective_blocking
      || _cm == CommunicationModel::collective_nonblocking)
    CHECK_MPI(MPI_Comm_free(&_neighbour_comm));
  if (_cm == CommunicationModel::p2p_blocking
      || _cm == CommunicationModel::p2p_nonblocking)
    delete[] _req;
  if (_cm == CommunicationModel::collective_nonblocking)
    delete _req;
  // FIXME: more cases
  if (_cm == CommunicationModel::p2p_nonblocking
      || _cm == CommunicationModel::collective_nonblocking) {
    _exec->free(_send_buf);
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_collective(T* vec_data) const
{
  assert(_cm == CommunicationModel::collective_blocking);

  // Allocate send buffer, if this is the first call to update
  if (_send_buf == nullptr) {
    if (_num_indices > 0) {
      _send_buf = _exec->alloc<T>(_num_indices);
    } else {
      // Send buffer cannot be null
      _send_buf = _exec->alloc<T>(1);
    }
  }

  // Get data from local indices to send to other processes, landing in their
  // ghost region
  _exec->gather_ghosts_run(_num_indices, _indexbuf, vec_data, (T*)_send_buf);

  // Send actual values - NB meaning of _send and _recv count/offset is
  // reversed
  MPI_Datatype data_type = mpi_type<T>();
  CHECK_MPI(MPI_Neighbor_alltoallv(
      _send_buf, _recv_count.data(), _recv_offset.data(), data_type, vec_data,
      _send_count.data(), _send_offset.data(), data_type, _neighbour_comm));
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_collective_start(T* vec_data) const
{
  assert(_cm == CommunicationModel::collective_nonblocking);

  // Allocate send buffer, if this is the first call to update
  if (_send_buf == nullptr) {
    if (_num_indices > 0) {
      _send_buf = _exec->alloc<T>(_num_indices);
    } else {
      // Send buffer cannot be null
      _send_buf = _exec->alloc<T>(1);
    }
  }

  // Get data from local indices to send to other processes, landing in their
  // ghost region
  _exec->gather_ghosts_run(_num_indices, _indexbuf, vec_data, (T*)_send_buf);

  // Send actual values - NB meaning of _send and _recv count/offset is
  // reversed
  MPI_Datatype data_type = mpi_type<T>();
  CHECK_MPI(MPI_Ineighbor_alltoallv(_send_buf, _recv_count.data(),
                                    _recv_offset.data(), data_type, vec_data,
                                    _send_count.data(), _send_offset.data(),
                                    data_type, _neighbour_comm, _req));
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_collective_end(T* vec_data) const
{
  assert(_cm == CommunicationModel::collective_nonblocking);
  CHECK_MPI(MPI_Wait(_req, MPI_STATUS_IGNORE));
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_p2p(T* vec_data) const
{
  assert(_cm == CommunicationModel::p2p_blocking);

  // Allocate send buffer, if this is the first call to update
  if (_send_buf == nullptr) {
    if (_num_indices > 0) {
      _send_buf = _exec->alloc<T>(_num_indices);
    } else {
      // Send buffer cannot be null
      _send_buf = _exec->alloc<T>(1);
    }
  }

  // Get data from local indices to send to other processes, landing in their
  // ghost region
  _exec->gather_ghosts_run(_num_indices, _indexbuf, vec_data, (T*)_send_buf);

  // Send actual values - NB meaning of _send and _recv count/offset is
  // reversed
  MPI_Datatype data_type = mpi_type<T>();
  const int num_neighbours = _neighbours.size();
  for (int i = 0; i < num_neighbours; ++i) {
    T* recv_buf = vec_data + _send_offset[i];
    CHECK_MPI(MPI_Irecv(recv_buf, _send_count[i], data_type, _neighbours[i], 0,
                        _comm, &(_req[i])));
  }

  for (int i = 0; i < num_neighbours; ++i) {
    T* send_buf = static_cast<T*>(_send_buf) + _recv_offset[i];
    CHECK_MPI(MPI_Send(send_buf, _recv_count[i], data_type, _neighbours[i], 0,
                       _comm));
  }

  CHECK_MPI(MPI_Waitall(num_neighbours, _req, MPI_STATUSES_IGNORE));
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_p2p_start(T* vec_data) const
{
  assert(_cm == CommunicationModel::p2p_nonblocking);

  // Allocate send buffer, if this is the first call to update
  if (_send_buf == nullptr) {
    if (_num_indices > 0) {
      _send_buf = _exec->alloc<T>(_num_indices);
    } else {
      // Send buffer cannot be null
      _send_buf = _exec->alloc<T>(1);
    }
  }

  // Get data from local indices to send to other processes, landing in their
  // ghost region
  _exec->gather_ghosts_run(_num_indices, _indexbuf, vec_data, (T*)_send_buf);

  // Send actual values - NB meaning of _send and _recv count/offset is
  // reversed
  MPI_Datatype data_type = mpi_type<T>();
  const int num_neighbours = _neighbours.size();
  for (int i = 0; i < num_neighbours; ++i) {
    T* recv_buf = vec_data + _send_offset[i];
    CHECK_MPI(MPI_Irecv(recv_buf, _send_count[i], data_type, _neighbours[i], 0,
                        _comm, &(_req[i])));
  }

  for (int i = 0; i < num_neighbours; ++i) {
    T* send_buf = static_cast<T*>(_send_buf) + _recv_offset[i];
    CHECK_MPI(MPI_Isend(send_buf, _recv_count[i], data_type, _neighbours[i], 0,
                        _comm, &(_req[num_neighbours + i])));
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_p2p_end(T* vec_data) const
{
  assert(_cm == CommunicationModel::p2p_nonblocking);
  CHECK_MPI(MPI_Waitall(2 * _neighbours.size(), _req, MPI_STATUSES_IGNORE));
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_onesided_put_active(T* vec_data) const
{
  assert(_cm == CommunicationModel::onesided_put_active);

  // Allocate send buffer, if this is the first call to update
  if (_send_buf == nullptr) {
    if (_num_indices > 0) {
      _send_buf = _exec->alloc<T>(_num_indices);
    } else {
      // Send buffer cannot be null
      _send_buf = _exec->alloc<T>(1);
    }
  }

  // Get data from local indices to send to other processes, landing in their
  // ghost region
  _exec->gather_ghosts_run(_num_indices, _indexbuf, vec_data, (T*)_send_buf);

  // Create remotely accesible window for vec_data
  MPI_Win win;
  CHECK_MPI(MPI_Win_create(vec_data + local_size(), num_ghosts() * sizeof(T),
                           sizeof(T), MPI_INFO_NULL, _comm, &win));

  // For every neighbor put the required data
  const int num_neighbours = _neighbours.size();
  MPI_Datatype data_type = mpi_type<T>();
  // Synchronize private and public windows
  CHECK_MPI(MPI_Win_fence(MPI_MODE_NOPRECEDE, win));
  for (int i = 0; i < num_neighbours; ++i) {
    T* send_buf = static_cast<T*>(_send_buf) + _recv_offset[i];
    CHECK_MPI(MPI_Put(send_buf, _recv_count[i], data_type, _neighbours[i],
                      _recv_win_offset[i], _recv_count[i], data_type, win));
  }
  // Synchronize private and public windows
  CHECK_MPI(MPI_Win_fence(MPI_MODE_NOSUCCEED, win));

  CHECK_MPI(MPI_Win_free(&win));
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_onesided_put_passive(T* vec_data) const
{
  assert(_cm == CommunicationModel::onesided_put_passive);

  // Allocate send buffer, if this is the first call to update
  if (_send_buf == nullptr) {
    if (_num_indices > 0) {
      _send_buf = _exec->alloc<T>(_num_indices);
    } else {
      // Send buffer cannot be null
      _send_buf = _exec->alloc<T>(1);
    }
  }

  // Get data from local indices to send to other processes, landing in their
  // ghost region
  _exec->gather_ghosts_run(_num_indices, _indexbuf, vec_data, (T*)_send_buf);

  // Create remotely accesible window for vec_data
  MPI_Win win;
  CHECK_MPI(MPI_Win_create(vec_data + local_size(), num_ghosts() * sizeof(T),
                           sizeof(T), MPI_INFO_NULL, _comm, &win));

  // For every neighbor put the required data
  const int num_neighbours = _neighbours.size();
  MPI_Datatype data_type = mpi_type<T>();
  for (int i = 0; i < num_neighbours; ++i) {
    T* send_buf = static_cast<T*>(_send_buf) + _recv_offset[i];
    CHECK_MPI(
        MPI_Win_lock(MPI_LOCK_SHARED, _neighbours[i], MPI_MODE_NOCHECK, win));
    CHECK_MPI(MPI_Put(send_buf, _recv_count[i], data_type, _neighbours[i],
                      _recv_win_offset[i], _recv_count[i], data_type, win));
    CHECK_MPI(MPI_Win_unlock(_neighbours[i], win));
  }

  CHECK_MPI(MPI_Win_free(&win));
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_shmem(T* vec_data) const
{
  assert(_cm == CommunicationModel::shmem);

  // In this approach, for on-node communication, instead of using MPI to
  // exchange data, we apply the same concept using shared memory windows

  // Create shared memory window for intra-node data exchange
  if (_window_mem == nullptr) {
    MPI_Aint alloc_len = std::accumulate(_recv_count_on_node.begin(),
                                         _recv_count_on_node.end(), 0);
    int disp_unit = sizeof(T);
    MPI_Info info;
    CHECK_MPI(MPI_Info_create(&info));
    // NUMA-aware allocation, prevents cache-line false sharing
    CHECK_MPI(MPI_Info_set(info, "alloc_shared_noncontig", "true"));
    CHECK_MPI(MPI_Win_allocate_shared(alloc_len * disp_unit, disp_unit, info,
                                      _node_comm, &_window_mem, &_window));
  }

  // Phase 1: gather values from vector into window and MPI buffer
  // Allocate send buffer, if this is the first call to update
  if (_send_buf == nullptr) {
    if (_num_indices > 0) {
      _send_buf = _exec->alloc<T>(_num_indices);
    } else {
      // Send buffer cannot be null
      _send_buf = _exec->alloc<T>(1);
    }
  }
  _exec->gather_ghosts_run(_num_indices, _indexbuf, vec_data, (T*)_send_buf);

  // Create a passive target epoch
  CHECK_MPI(MPI_Win_lock_all(MPI_MODE_NOCHECK, _window));
  int cnt = 0;
  T* window_ptr = static_cast<T*>(_window_mem);
  for (auto rc : _receivers_on_node) {
    for (int j = 0; j < _recv_count_on_node[rc]; ++j) {
      // FIXME: Shouldn't this be similar to _indexbuf?
      window_ptr[cnt + j]
          = vec_data[_indexbuf_on_node[_recv_offset_on_node[rc] + j]];
    }
    cnt += _recv_count_on_node[rc];
  }

  // Sychronize private and public copies
  // Flush my local changes (cache) to main memory
  CHECK_MPI(MPI_Win_sync(
      _window)); // atomic_thread_fence(memory_order_leasease/acquire);
  // Sync processes
  CHECK_MPI(MPI_Barrier(_node_comm));
  // Sync my local memory (cache) with changes in main memory
  CHECK_MPI(MPI_Win_sync(
      _window)); // atomic_thread_fence(memory_order_leasease/acquire);

  // Remote load epoch can start
  // Gather the ghosts I need from my neighbours' window
  for (auto s : _senders_on_node) {
    T* window_ptr = nullptr;
    int disp_unit;
    MPI_Aint alloc_len;
    CHECK_MPI(
        MPI_Win_shared_query(_window, s, &alloc_len, &disp_unit, &window_ptr));
    T* ptr = window_ptr + _sender_window_offset[s];
    memcpy(vec_data + _send_offset_on_node[s], ptr,
           _send_count_on_node[s] * sizeof(T));
  }
  CHECK_MPI(MPI_Win_unlock_all(_window));

  MPI_Datatype data_type = mpi_type<T>();
  CHECK_MPI(MPI_Neighbor_alltoallv(
      _send_buf, _recv_count.data(), _recv_offset.data(), data_type, vec_data,
      _send_count.data(), _send_offset.data(), data_type, _neighbour_comm));
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_shmem_nodup(T* vec_data) const
{
  assert(_cm == CommunicationModel::shmem);

  // In this approach, for on-node communication, instead of duplicating data
  // depending on how many neighbors require each vector value, we instead
  // expose the vectors in a shared memory window and let each process get
  // the values it needs
  // Create shared memory window for intra-node data exchange
  if (_window_mem == nullptr) {
    MPI_Aint alloc_len = local_size();
    int disp_unit = sizeof(T);
    MPI_Info info;
    CHECK_MPI(MPI_Info_create(&info));
    // NUMA-aware allocation, prevents cache-line false sharing
    CHECK_MPI(MPI_Info_set(info, "alloc_shared_noncontig", "true"));
    CHECK_MPI(MPI_Win_allocate_shared(alloc_len * disp_unit, disp_unit, info,
                                      _node_comm, &_window_mem, &_window));
  }

  // Phase 1: copy local vector to window
  // Create a passive target epoch
  CHECK_MPI(MPI_Win_lock_all(MPI_MODE_NOCHECK, _window));
  memcpy(_window_mem, vec_data, local_size() * sizeof(T));

  // Sychronize private and public copies
  // Flush my local changes (cache) to main memory
  CHECK_MPI(MPI_Win_sync(
      _window)); // atomic_thread_fence(memory_order_leasease/acquire);
  // Sync processes
  CHECK_MPI(MPI_Barrier(_node_comm));
  // Sync my local memory (cache) with changes in main memory
  CHECK_MPI(MPI_Win_sync(
      _window)); // atomic_thread_fence(memory_order_leasease/acquire);

  // Remote load epoch can start
  // Gather the ghosts I need from my neighbours' window
  for (auto s : _senders_on_node) {
    T* window_ptr = nullptr;
    int disp_unit;
    MPI_Aint alloc_len;
    CHECK_MPI(
        MPI_Win_shared_query(_window, s, &alloc_len, &disp_unit, &window_ptr));
    int window_offset = _send_offset_on_node[s] - local_size();
    for (int i = 0; i < _send_count_on_node[s]; i++) {
      vec_data[_send_offset_on_node[s] + i]
          = window_ptr[ghost_local_indices_on_node[window_offset + i]];
    }
  }
  CHECK_MPI(MPI_Win_unlock_all(_window));

  if (_send_buf == nullptr) {
    if (_num_indices > 0) {
      _send_buf = _exec->alloc<T>(_num_indices);
    } else {
      // Send buffer cannot be null
      _send_buf = _exec->alloc<T>(1);
    }
  }
  _exec->gather_ghosts_run(_num_indices, _indexbuf, vec_data, (T*)_send_buf);

  MPI_Datatype data_type = mpi_type<T>();
  CHECK_MPI(MPI_Neighbor_alltoallv(
      _send_buf, _recv_count.data(), _recv_offset.data(), data_type, vec_data,
      _send_count.data(), _send_offset.data(), data_type, _neighbour_comm));
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update(T* vec_data) const
{
  switch (_cm) {
  case CommunicationModel::p2p_blocking:
    update_p2p(vec_data);
    break;
  case CommunicationModel::p2p_nonblocking:
    update_p2p_start(vec_data);
    break;
  case CommunicationModel::collective_blocking:
    update_collective(vec_data);
    break;
  case CommunicationModel::collective_nonblocking:
    update_collective_start(vec_data);
    break;
  case CommunicationModel::onesided_put_active:
    update_onesided_put_active(vec_data);
    break;
  case CommunicationModel::onesided_put_passive:
    update_onesided_put_passive(vec_data);
    break;
  case CommunicationModel::shmem:
    update_shmem(vec_data);
    break;
  case CommunicationModel::shmem_nodup:
    update_shmem_nodup(vec_data);
    break;
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::update_finalise(T* vec_data) const
{
  if (_cm == CommunicationModel::p2p_nonblocking)
    update_p2p_end(vec_data);
  else if (_cm == CommunicationModel::collective_nonblocking)
    update_collective_end(vec_data);
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::reverse_update_collective(T* vec_data) const
{

  // Send values from ghost region of vector to remotes
  // accumulating in local vector.
  std::vector<T> databuf(_num_indices);
  MPI_Datatype data_type = mpi_type<T>();
  CHECK_MPI(
      MPI_Neighbor_alltoallv(vec_data, _send_count.data(), _send_offset.data(),
                             data_type, databuf.data(), _recv_count.data(),
                             _recv_offset.data(), data_type, _neighbour_comm));

  for (int i = 0; i < _num_indices; ++i)
    vec_data[_indexbuf[i]] += databuf[i];
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::reverse_update_p2p(T* vec_data) const
{
  // Send values from ghost region of vector to remotes
  // accumulating in local vector.
  std::vector<T> databuf(_num_indices);
  MPI_Datatype data_type = mpi_type<T>();
  const int num_neighbours = _neighbours.size();
  MPI_Request* rq = _req;

  for (int i = 0; i < num_neighbours; ++i) {
    T* recv_buf = databuf.data() + _recv_offset[i];
    CHECK_MPI(MPI_Irecv(recv_buf, _recv_count[i], data_type, _neighbours[i], 0,
                        _comm, rq++));
  }

  for (int i = 0; i < num_neighbours; ++i) {
    T* send_buf = vec_data + _send_offset[i];
    CHECK_MPI(MPI_Isend(send_buf, _send_count[i], data_type, _neighbours[i], 0,
                        _comm, rq++));
  }

  CHECK_MPI(MPI_Waitall(num_neighbours, _req, MPI_STATUS_IGNORE));

  for (int i = 0; i < _num_indices; ++i)
    vec_data[_indexbuf[i]] += databuf[i];
}
//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::reverse_update(T* vec_data) const
{
  if (_cm == CommunicationModel::p2p_blocking)
    reverse_update_p2p(vec_data);
  else if (_cm == CommunicationModel::collective_blocking)
    reverse_update_collective(vec_data);
}
//-----------------------------------------------------------------------------
std::int32_t L2GMap::global_to_local(std::int64_t i) const
{
  const std::int64_t r0 = _ranges[_mpi_rank];
  const std::int64_t r1 = _ranges[_mpi_rank + 1];

  if (i >= r0 and i < r1)
    return (i - r0);
  else {
    auto it = _global_to_local.find(i);
    assert(it != _global_to_local.end());
    return it->second;
  }
}
//-----------------------------------------------------------------------------
bool L2GMap::overlapping() const
{
  return (_cm == CommunicationModel::p2p_nonblocking
          || _cm == CommunicationModel::collective_nonblocking)
             ? true
             : false;
}
//-----------------------------------------------------------------------------
MPI_Comm L2GMap::global_comm() const { return _comm; }
//-----------------------------------------------------------------------------
int L2GMap::rank() const { return _mpi_rank; }
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
// template void
// spmv::L2GMap::update<std::complex<float>>(std::complex<float>*) const;
// template void
// spmv::L2GMap::update<std::complex<double>>(std::complex<double>*) const;
template void spmv::L2GMap::update_finalise<double>(double*) const;
template void spmv::L2GMap::update_finalise<float>(float*) const;
// template void
// spmv::L2GMap::update_finalise<std::complex<float>>(std::complex<float>*)
// const; template void spmv::L2GMap::update_finalise<std::complex<double>>(
//     std::complex<double>*) const;
template void spmv::L2GMap::reverse_update<double>(double*) const;
template void spmv::L2GMap::reverse_update<float>(float*) const;
// template void
// spmv::L2GMap::reverse_update<std::complex<float>>(std::complex<float>*)
// const; template void
// spmv::L2GMap::reverse_update<std::complex<double>>(std::complex<double>*)
// const;

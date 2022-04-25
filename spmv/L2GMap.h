// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include "config.h"
#include "spmv_export.h"

#include "mpi_utils.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

namespace spmv
{

// Forward declarations
class DeviceExecutor;

class SPMV_EXPORT L2GMap
/// @brief Local to Global Map
///
/// Maps from the local indices on the current process to global indices across
/// all processes. The local process owns a contiguous set of the global
/// indices, starting at "global_offset". Any indices which are not owned appear
/// as "ghost entries" at the end of the local range.
{
public:
  /// L2GMap (Local to Global Map)
  /// ----------------------------
  /// @param comm MPI Comm
  /// @param local_size Local size
  /// @param ghosts Ghost indices, owned by other processes.
  /// @param p2p Use point-to-point communication.
  /// @param overlap Overlap communication with computation.
  /// Ghosts must be sorted in ascending order.
  L2GMap(MPI_Comm comm, std::int64_t local_size,
         const std::vector<std::int64_t>& ghosts,
         std::shared_ptr<spmv::DeviceExecutor> exec,
         CommunicationModel cm = CommunicationModel::collective_blocking);

  // Destructor destroys neighbour comm
  ~L2GMap();

  // Disable copying (may cause problems with held neighbour comm)
  L2GMap(const L2GMap& p) = delete;
  L2GMap& operator=(const L2GMap& p) = delete;

  /// Local size
  /// @return The local size, not including ghost entries.
  std::int32_t local_size() const;

  /// Number of ghost entries
  /// @return The number of ghost entries held locally.
  std::int32_t num_ghosts() const;

  /// Global size
  /// @return global size of L2GMap
  std::int64_t global_size() const;

  /// Global offset on this process
  /// @return Global index of first local index
  std::int64_t global_offset() const;

  /// Convert a global index to local
  /// @param i Global Index
  /// @return Local index
  std::int32_t global_to_local(std::int64_t i) const;

  /// Overlapping
  /// @return A flag indicating whether comp/comm ovelap is enabled
  bool overlapping() const;

  /// Global MPI communicator
  /// @return The global MPI communicator.
  MPI_Comm global_comm() const;

  /// Global MPI rank
  /// @return The global MPI rank.
  int rank() const;

  /// Ghost update. Copies values from remote indices to the local process.
  /// This should be applied to a vector *before* a MatVec operation, if the
  /// Matrix has column ghosts.
  /// @param vec_data Pointer to vector data
  template <typename T>
  void update(T* vec_data) const;

  /// Ghost update finalisation. Completes MPI communication. This should be
  /// called when ovelapping is enabled.
  template <typename T>
  void update_finalise(T* vec_data) const;

  /// Reverse update. Sends ghost values to their owners, where they are
  /// accumulated at the local index. This should be applied to the result
  /// *after* a MatVec operation, if the Matrix has row ghosts.
  /// @param vec_data Pointer to vector data
  template <typename T>
  void reverse_update(T* vec_data) const;

  /// Access the ghost indices
  const std::vector<std::int64_t>& ghosts() const { return _ghosts; }

private:
  // Device executor
  std::shared_ptr<spmv::DeviceExecutor> _exec;

  // Ownership ranges for all processes on global communicator
  std::vector<std::int64_t> _ranges = {};

  // Cached mpi rank on global communicator
  // Local range is _ranges[_mpi_rank] -> _ranges[_mpi_rank + 1]
  std::int32_t _mpi_rank = -1;
  // Cached mpi rank on node communicator
  std::int32_t _mpi_node_rank = -1;

  // Forward and reverse maps for ghosts
  std::map<std::int64_t, std::int32_t> _global_to_local = {};
  std::vector<std::int64_t> _ghosts = {};

  // Indices, counts and offsets for communication
  int _num_indices = 0;
  int* _indexbuf = nullptr;
  std::vector<std::int32_t> _send_count = {};
  std::vector<std::int32_t> _recv_count = {};
  std::vector<std::int32_t> _send_offset = {};
  std::vector<std::int32_t> _recv_offset = {};
  std::vector<std::int32_t> _recv_win_offset = {};

  // On-node communication
  // Node ranks I need to get data from
  std::vector<std::int32_t> _senders_on_node = {};
  // Node ranks I need to send data to
  std::vector<std::int32_t> _receivers_on_node = {};
  // Indices, counts and offsets
  std::vector<std::int32_t> _indexbuf_on_node = {};
  std::vector<std::int32_t> _send_count_on_node = {};
  std::vector<std::int32_t> _recv_count_on_node = {};
  std::vector<std::int32_t> _send_offset_on_node = {};
  std::vector<std::int32_t> _recv_offset_on_node = {};
  // Where in the window I need to write to
  std::vector<std::int32_t> _receiver_window_offset = {};
  // Where in the window I need to write to
  std::vector<std::int32_t> _sender_window_offset = {};
  std::vector<std::int32_t> ghost_local_indices_on_node = {};
  std::vector<std::int32_t> _unique_indices = {};
  // Shared-memory window for on-node communication
  mutable MPI_Win _window;
  mutable void* _window_mem = nullptr;

  // Ranks of my neighbours
  std::vector<int> _neighbours = {};
  // Global communicator
  MPI_Comm _comm = MPI_COMM_NULL;
  // Node-level communication
  MPI_Comm _node_comm = MPI_COMM_NULL;
  // Neighbourhood communicator
  MPI_Comm _neighbour_comm = MPI_COMM_NULL;
  // Underlying MPI comunnication model
  CommunicationModel _cm = CommunicationModel::collective_blocking;
  // MPI handle and intermediate buffers used to manage non-blocking
  // communication
  mutable MPI_Request* _req = nullptr;
  mutable void* _send_buf = nullptr;

private:
  // Private functions
  template <typename T>
  void update_collective(T* vec_data) const;
  template <typename T>
  void update_collective_start(T* vec_data) const;
  template <typename T>
  void update_collective_end(T* vec_data) const;
  template <typename T>
  void update_p2p(T* vec_data) const;
  template <typename T>
  void update_p2p_start(T* vec_data) const;
  template <typename T>
  void update_p2p_end(T* vec_data) const;
  template <typename T>
  void update_onesided_put_active(T* vec_data) const;
  template <typename T>
  void update_onesided_put_passive(T* vec_data) const;
  template <typename T>
  void update_shmem(T* vec_data) const;
  template <typename T>
  void update_shmem_nodup(T* vec_data) const;
  template <typename T>
  void reverse_update_collective(T* vec_data) const;
  template <typename T>
  void reverse_update_p2p(T* vec_data) const;
};

} // namespace spmv

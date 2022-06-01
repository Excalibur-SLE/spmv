// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include "config.h"
#include "spmv_export.h"

#include <memory>
#include <mpi.h>
#include <string>

#include "mpi_utils.h"

namespace spmv
{

// Forward declarations
class L2GMap;
class DeviceExecutor;

template <typename>
class Matrix;

/// @brief Read a binary PETSc matrix file (32-bit indices).
///
/// Create a suitable file with petsc option "-ksp_view_mat binary"
/// @param comm MPI Comm
/// @param filename Filename
/// @param symmetric Indicates whether the matrix is symmetric
/// @return spmv::Matrix<double> Matrix
SPMV_EXPORT
Matrix<double> read_petsc_binary_matrix(
    std::string filename, MPI_Comm comm, std::shared_ptr<DeviceExecutor> exec,
    bool symmetric = false,
    CommunicationModel cm = CommunicationModel::collective_blocking);

/// @brief Read a binary PETSc vector file and distribute.
///
/// Create a suitable file with petsc option "-ksp_view_rhs binary"
/// @param comm MPI Communicator
/// @param filename Filename
/// @return Vector of values
SPMV_EXPORT
double* read_petsc_binary_vector(MPI_Comm comm, const DeviceExecutor* exec,
                                 const std::string filename);
} // namespace spmv

// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>
#include <mpi.h>
#include <string>

#include "mpi_types.h"

#pragma once

namespace spmv
{

// Forward declarations
class L2GMap;

template <typename>
class Matrix;

/// @brief Read a binary PETSc matrix file (32-bit indices).
///
/// Create a suitable file with petsc option "-ksp_view_mat binary"
/// @param comm MPI Comm
/// @param filename Filename
/// @param symmetric Indicates whether the matrix is symmetric
/// @return spmv::Matrix<double> Matrix
#ifdef USE_CUDA
Matrix<double> read_petsc_binary_matrix(MPI_Comm comm, std::string filename,
                                        bool symmetric = false,
                                        CommunicationModel cm
                                        = CommunicationModel::p2p_blocking);
#else
Matrix<double> read_petsc_binary_matrix(
    MPI_Comm comm, std::string filename, bool symmetric = false,
    CommunicationModel cm = CommunicationModel::collective_blocking);
#endif

/// @brief Read a binary PETSc vector file and distribute.
///
/// Create a suitable file with petsc option "-ksp_view_rhs binary"
/// @param comm MPI Communicator
/// @param filename Filename
/// @return Vector of values
Eigen::VectorXd read_petsc_binary_vector(MPI_Comm comm, std::string filename);
} // namespace spmv

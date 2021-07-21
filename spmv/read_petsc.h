// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <Eigen/Sparse>
#include <memory>
#include <mpi.h>
#include <string>

#include "Matrix.h"

#pragma once

namespace spmv
{
class L2GMap;

/// @brief Read a binary PETSc matrix file (32-bit indices).
///
/// Create a suitable file with petsc option "-ksp_view_mat binary"
/// @param comm MPI Comm
/// @param filename Filename
/// @param symmetric Indicates whether the matrix is symmetric
/// @return spmv::Matrix<double> Matrix
Matrix<double> read_petsc_binary(MPI_Comm comm, std::string filename,
                                 bool symmetric = false,
                                 CommunicationModel cm
                                 = CommunicationModel::collective_blocking);

/// @brief Read a binary PETSc vector file and distribute.
///
/// Create a suitable file with petsc option "-ksp_view_rhs binary"
/// @param comm MPI Communicator
/// @param filename Filename
/// @return Vector of values
Eigen::VectorXd read_petsc_binary_vector(MPI_Comm comm, std::string filename);
} // namespace spmv

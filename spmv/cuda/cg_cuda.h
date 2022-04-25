// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include "config.h"
#include "spmv_export.h"

#include <mpi.h>

namespace spmv
{

// Forward declarations
class CudaExecutor;
template <typename T>
class Matrix;

/// @brief Solve **A.x=b** iteratively with Conjugate Gradient
///
/// Input
/// @param comm MPI communicator
/// @param A LHS matrix
/// @param b RHS vector
/// @param max_its Maximum iteration count
/// @param rtol Relative tolerance
///
/// @return tuple of result **x** and number of iterations
///
SPMV_EXPORT
int cg(MPI_Comm comm, CudaExecutor& exec, const Matrix<double>& A,
       const double* b, double* x, int max_its, double rtol);

} // namespace spmv

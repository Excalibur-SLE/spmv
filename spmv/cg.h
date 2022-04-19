// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once
#include "spmv_export.h"

#include <mpi.h>

namespace spmv
{

// Forward declarations
template <typename T>
class Matrix;
class ReferenceExecutor;

/// @brief Solve **A.x=b** iteratively with Conjugate Gradient
///
/// Input
/// @param comm MPI communicator
/// @param exec Device executor
/// @param A LHS matrix
/// @param b RHS vector
/// @param x LHS vector
/// @param max_its Maximum iteration count
/// @param rtol Relative tolerance
///
/// @return number of iterations
///
SPMV_EXPORT int cg(MPI_Comm comm, ReferenceExecutor& exec,
                   const spmv::Matrix<double>& A, const double* b, double* x,
                   int kmax, double rtol);

} // namespace spmv

// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include <mpi.h>

namespace spmv
{

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
std::tuple<double*, int> cg(MPI_Comm comm, const Matrix<double>& A, double* b,
                            int max_its, double rtol);

} // namespace spmv

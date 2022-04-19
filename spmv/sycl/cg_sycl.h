// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once
#include <CL/sycl.hpp>
#include <mpi.h>

namespace sycl = cl::sycl;

namespace spmv
{

template <typename T>
class Matrix;
class SyclExecutor;

/// @brief Solve **A.x=b** iteratively with Conjugate Gradient in SYCL
///
/// Input
/// @param comm MPI communicator
/// @param A LHS matrix
/// @param b RHS vector as a USM pointer
/// @param max_its Maximum iteration count
/// @param rtol Relative tolerance
///
/// @return tuple of result **x** as USM pointer and number of iterations
///
SPMV_EXPORT
int cg(MPI_Comm comm, SyclExecutor& exec, const spmv::Matrix<double>& A,
       const double* b, double* x, int kmax, double rtol);
} // namespace spmv

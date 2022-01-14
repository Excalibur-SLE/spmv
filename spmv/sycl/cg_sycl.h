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

/// @brief Solve **A.x=b** iteratively with Conjugate Gradient in SYCL
///
/// Input
/// @param comm MPI communicator
/// @param A LHS matrix
/// @param b RHS vector as a SYCL buffer
/// @param max_its Maximum iteration count
/// @param rtol Relative tolerance
///
/// @return tuple of result **x** as host pointer and number of iterations
///
SPMV_EXPORT
std::tuple<double*, int> cg(MPI_Comm comm, const spmv::Matrix<double>& A,
                            sycl::buffer<double>& b_buf, int kmax, double rtol,
                            sycl::queue& queue);

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
std::tuple<double*, int> cg(MPI_Comm comm, const spmv::Matrix<double>& A,
                            double* b, int kmax, double rtol,
                            sycl::queue& queue);
} // namespace spmv

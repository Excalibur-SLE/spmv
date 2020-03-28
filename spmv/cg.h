// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// SPDX-License-Identifier:    MIT

#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
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
std::tuple<Eigen::VectorXd, int> cg(MPI_Comm comm, const Matrix<double>& A,
                                    const Eigen::Ref<const Eigen::VectorXd>& b,
                                    int max_its, double rtol);

} // namespace spmv

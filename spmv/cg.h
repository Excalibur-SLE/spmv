// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <mpi.h>

#ifdef _SYCL
#include <CL/sycl.hpp>
namespace sycl = sycl;
#endif

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

#ifdef _SYCL
/// @brief Compute vector r = alpha*x + y
///
/// Input
/// @param q SYCL queue
/// @param N Length of vectors
/// @param alpha Scalar
/// @param x Input vector
/// @param y Input vector
///
/// Output
/// @param r Output vector
template <typename T>
sycl::event axpy(sycl::queue& q, size_t N, T alpha, const T* x, const T* y,
                 T* r)
{
  sycl::event event = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class aXpY>(sycl::range<1>{N}, [=](sycl::item<1> it) {
      int i = it.get_id(0);
      r[i] = alpha * x[i] + y[i];
    });
  });

  return event;
}

/// @brief Compute the dot product of two vectors
///
/// Input
/// @param q SYCL queue
/// @param N Length of vectors
/// @param x Input vector
/// @param y Input vector
///
/// Output
/// @return Dot product of vectors x and y
template <typename T>
T dot(sycl::queue& q, size_t N, const T* x, const T* y)
{
  if constexpr (std::is_same<T, std::complex<double>>::value
                or std::is_same<T, std::complex<float>>::value)
    throw std::runtime_error("Complex support");

  auto sum = sycl::malloc_shared<T>(1, q);
  sum[0] = 0.0;

  const int M = 32;
  const int mod = N % M;
  const int N_padded = (mod == 0) ? N : N + M - mod;

  // Compute a dot-product by reducing all computed values using standard plus
  // functor
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class dot_product>(
        sycl::nd_range<1>{N_padded, M},
        sycl::ONEAPI::reduction(sum, 0.0, std::plus<T>()),
        [=](sycl::nd_item<1> it, auto& sum) {
          int i = it.get_global_id(0);
          sum += (i < N_padded) ? (x[i] * y[i]) : 0.0;
        });
  });
  q.wait();

  T s = sum[0];
  sycl::free(sum, q);
  return s;
}

template <typename T>
double squared_norm(sycl::queue& q, size_t N, const T* x)
{
  return dot(q, N, x, x);
}

template <typename T>
sycl::event fused_update(sycl::queue& q, size_t N, T alpha, const T* p, T* x,
                         const T* y, T* r)
{
  sycl::event event = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class update>(sycl::range<1>{N}, [=](sycl::item<1> it) {
      int i = it.get_id(0);
      x[i] += alpha * p[i];
      r[i] += -alpha * y[i];
    });
  });

  return event;
}

/// @brief Solve **A.x=b** iteratively with Conjugate Gradient in SYCL
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
std::tuple<double*, int> cg_sycl(MPI_Comm comm, sycl::queue& queue,
                                 const spmv::Matrix<double>& A, double* b,
                                 int kmax, double rtol);
#endif

} // namespace spmv

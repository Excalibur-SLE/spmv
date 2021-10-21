// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once
#include <CL/sycl.hpp>
#include <complex>
#include <mpi.h>

namespace sycl = cl::sycl;

namespace spmv
{

template <typename T>
class Matrix;

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
sycl::event axpy(sycl::queue& queue, size_t N, T alpha, const T* x, const T* y,
                 T* r, const std::vector<sycl::event>& dependencies = {})
{
  sycl::event event = queue.submit(
      [&](sycl::handler& cgh)
      {
        cgh.depends_on(dependencies);
        cgh.parallel_for<>(sycl::range<1>{N},
                           [=](sycl::item<1> item) {
                             int i = item.get_id(0);
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
sycl::event dot(sycl::queue& queue, size_t N, const T* x, const T* y, T* result,
                const std::vector<sycl::event>& dependencies = {})
{
  if constexpr (std::is_same<T, std::complex<double>>::value
                or std::is_same<T, std::complex<float>>::value)
    throw std::runtime_error("Complex support");

  sycl::event event;
  auto sum = sycl::malloc_shared<T>(1, queue);

  // Compute a dot-product by reducing all computed values using standard plus
  // functor
  auto device = queue.get_device();
  if (device.is_gpu())
  {
    const int M = 32;
    const int mod = N % M;
    const int N_padded = (mod == 0) ? N : N + M - mod;

    // Compute a dot-product by reducing all computed values using standard plus
    // functor
    queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
	cgh.parallel_for<>(sycl::nd_range<1>{N_padded, M},
			   sycl::reduction(sum, std::plus<T>()),
			   [=](sycl::nd_item<1> item, auto& sum) {
			     int i = item.get_global_id(0);
			     sum.combine((i < N_padded) ? (x[i] * y[i]) : 0.0);
			   });
      }).wait_and_throw();
  }
  else
  {
    queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
	cgh.parallel_for<>(sycl::nd_range<1>{N, 1},
			   sycl::reduction(sum, std::plus<T>()),
			   [=](sycl::nd_item<1> item, auto& sum) {
			     int i = item.get_global_id(0);
			     sum.combine(x[i] * y[i]);
			   });
      }).wait_and_throw();
  }

  *result = sum[0];
  sycl::free(sum, queue);
  return event;
}

template <typename T>
sycl::event squared_norm(sycl::queue& queue, size_t N, const T* x, T* result,
                         const std::vector<sycl::event>& dependencies = {})
{
  return dot(queue, N, x, x, result, dependencies);
}

template <typename T>
sycl::event fused_update(sycl::queue& queue, size_t N, T alpha, const T* p,
                         T* x, const T* y, T* r,
                         const std::vector<sycl::event>& dependencies = {})
{
  sycl::event event = queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.parallel_for<>(sycl::range<1>{N},
			 [=](sycl::item<1> it) {
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
std::tuple<double*, int> cg(MPI_Comm comm, sycl::queue& queue,
                            const spmv::Matrix<double>& A, double* b, int kmax,
                            double rtol);

} // namespace spmv

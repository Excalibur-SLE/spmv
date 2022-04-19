// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once
#include <CL/sycl.hpp>
#include <complex>

#ifdef _DPCPP
#include "oneapi/mkl.hpp"
#else
#include "blas_sycl.h"
#endif

namespace sycl = cl::sycl;

namespace spmv
{

/// @brief Compute vector r = alpha*x + y
///
/// Input
/// @param N Length of vectors
/// @param alpha Scalar
/// @param x Input vector
/// @param y Input vector
/// @param q SYCL queue
///
/// Output
/// @param r Output vector
template <typename T>
sycl::event axpy(size_t N, T alpha, const T* x, T* y, sycl::queue& queue,
                 const std::vector<sycl::event>& dependencies = {})
{
#ifdef _DPCPP
  return oneapi::mkl::blas::row_major::axpy(queue, N, alpha, x, 1, y, 1,
                                            dependencies);
#else
  sycl::event event = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dependencies);
    cgh.parallel_for<>(sycl::range<1>{N}, [=](sycl::item<1> item) {
      int i = item.get_id(0);
      y[i] = alpha * x[i] + y[i];
    });
  });
  return event;
#endif
}

/// @brief Compute the dot product of two vectors
///
/// Input
/// @param N Length of vectors
/// @param x Input vector
/// @param y Input vector
/// @param q SYCL queue
///
/// Output
/// @return Dot product of vectors x and y
template <typename T>
sycl::event dot(size_t N, const T* x, const T* y, T* result, sycl::queue& queue,
                const std::vector<sycl::event>& dependencies = {})
{
#ifdef _DPCPP
  return oneapi::mkl::blas::row_major::dot(queue, N, x, 1, y, 1, result);
#else
  if constexpr (std::is_same<T, std::complex<double>>::value
                or std::is_same<T, std::complex<float>>::value)
    throw std::runtime_error("Complex support");

  sycl::event event;
  auto sum = sycl::malloc_shared<T>(1, queue);

  // Compute a dot-product by reducing all computed values using standard plus
  // functor
  auto device = queue.get_device();
  if (device.is_gpu()) {
    const int M = 32;
    const int mod = N % M;
    const int N_padded = (mod == 0) ? N : N + M - mod;

    // Compute a dot-product by reducing all computed values using standard plus
    // functor
    queue
        .submit([&](sycl::handler& cgh) {
          cgh.depends_on(dependencies);
          cgh.parallel_for<>(sycl::nd_range<1>{N_padded, M},
                             sycl::reduction(sum, std::plus<T>()),
                             [=](sycl::nd_item<1> item, auto& sum) {
                               int i = item.get_global_id(0);
                               sum.combine((i < N_padded) ? (x[i] * y[i])
                                                          : 0.0);
                             });
        })
        .wait_and_throw();
  } else {
    queue
        .submit([&](sycl::handler& cgh) {
          cgh.depends_on(dependencies);
          cgh.parallel_for<>(sycl::nd_range<1>{N, 1},
                             sycl::reduction(sum, std::plus<T>()),
                             [=](sycl::nd_item<1> item, auto& sum) {
                               int i = item.get_global_id(0);
                               sum.combine(x[i] * y[i]);
                             });
        })
        .wait_and_throw();
  }

  *result = sum[0];
  sycl::free(sum, queue);
  return event;
#endif
}

template <typename T>
sycl::event squared_norm(size_t N, const T* x, T* result, sycl::queue& queue,
                         const std::vector<sycl::event>& dependencies = {})
{
  return dot(N, x, x, result, queue, dependencies);
}

template <typename T>
sycl::event fused_update(size_t N, T alpha, const T* p, T* x, const T* y, T* r,
                         sycl::queue& queue,
                         const std::vector<sycl::event>& dependencies = {})
{
  sycl::event event = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dependencies);
    cgh.parallel_for<>(sycl::range<1>{N}, [=](sycl::item<1> it) {
      int i = it.get_id(0);
      x[i] += alpha * p[i];
      r[i] += -alpha * y[i];
    });
  });

  return event;
}
} // namespace spmv

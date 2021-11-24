// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <complex>
#include <mpi.h>

#pragma once

namespace spmv
{
/// Obtain the MPI datatype for a given scalar type
template <typename T>
inline MPI_Datatype mpi_type();
// @cond
template <>
inline MPI_Datatype mpi_type<float>()
{
  return MPI_FLOAT;
}
template <>
inline MPI_Datatype mpi_type<std::complex<float>>()
{
  return MPI_C_FLOAT_COMPLEX;
}
template <>
inline MPI_Datatype mpi_type<double>()
{
  return MPI_DOUBLE;
}
template <>
inline MPI_Datatype mpi_type<std::complex<double>>()
{
  return MPI_DOUBLE_COMPLEX;
}
// @endcond

// MPI communication model
enum class CommunicationModel {
  p2p_blocking,
  p2p_nonblocking,
  collective_blocking,
  collective_nonblocking,
  onesided_put_active,
  onesided_put_passive
};

} // namespace spmv

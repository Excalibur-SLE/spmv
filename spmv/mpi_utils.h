// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <complex>
#include <mpi.h>

#ifdef _CUDA
#include <mpi-ext.h>
#if !defined(MPIX_CUDA_AWARE_SUPPORT)
#error "The MPI Implementation does not have CUDA-aware support or \
CUDA-aware support can't be determined."
#endif
#endif // _CUDA

#pragma once

namespace spmv
{

#define CHECK_MPI(func)                                                        \
  {                                                                            \
    int mpi_status = (func);                                                   \
    if (mpi_status != MPI_SUCCESS) {                                           \
      char mpi_error_string[MPI_MAX_ERROR_STRING];                             \
      int mpi_error_string_length = 0;                                         \
      MPI_Error_string(mpi_status, mpi_error_string,                           \
                       &mpi_error_string_length);                              \
      if (mpi_error_string != NULL)                                            \
        fprintf(stderr,                                                        \
                "ERROR: MPI call \"%s\" at line %d of file %s with %s (%d)\n", \
                #func, __LINE__, __FILE__, mpi_error_string, mpi_status);      \
      else                                                                     \
        fprintf(                                                               \
            stderr,                                                            \
            "ERROR: MPI call \"%s\" at line %d of file %s failed with %d.\n",  \
            #func, __LINE__, __FILE__, mpi_status);                            \
      exit(mpi_status);                                                        \
    }                                                                          \
  }

// MPI communication model
enum class CommunicationModel {
  p2p_blocking,
  p2p_nonblocking,
  collective_blocking,
  collective_nonblocking,
  onesided_put_active,
  onesided_put_passive,
  shmem,
  shmem_nodup
};

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

} // namespace spmv

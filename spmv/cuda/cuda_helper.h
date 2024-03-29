// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr,                                                          \
              "ERROR: CUDA API \"%s\" failed at line %d of file %s with %s "   \
              "(%d)\n",                                                        \
              #func, __LINE__, __FILE__, cudaGetErrorString(status), status);  \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      fprintf(stderr,                                                          \
              "ERROR: cuSPARSE API \"%s\" failed at line %d of file %s with "  \
              "%s (%d)\n",                                                     \
              #func, __LINE__, __FILE__, cusparseGetErrorString(status),       \
              status);                                                         \
    }                                                                          \
  }

// cuBLAS API errors
static const char* cublasGetErrorString(cublasStatus_t error)
{
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";

  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";

  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}

#define CHECK_CUBLAS(func)                                                     \
  {                                                                            \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr,                                                          \
              "ERROR: cuBLAS API \"%s\" failed at line %d of file %s with %s " \
              "(%d)\n",                                                        \
              #func, __LINE__, __FILE__, cublasGetErrorString(status),         \
              status);                                                         \
    }                                                                          \
  }

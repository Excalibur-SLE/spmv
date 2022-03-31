// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "config.h"

#include "L2GMap.h"
#include "Matrix.h"
#include "cg.h"
#include "device_executor.h"
#include "reference_executor.h"
#ifdef _OPENMP_HOST
#include "openmp/omp_executor.h"
#endif
#ifdef _OPENMP_OFFLOAD
#include "openmp_offload/omp_offload_executor.h"
#endif
#ifdef _CUDA
#include "cuda/cg_cuda.h"
#include "cuda/cuda_executor.h"
#endif
#ifdef _SYCL
#include "sycl/cg_sycl.h"
#include "sycl/sycl_executor.h"
#endif
#include "read_petsc.h"

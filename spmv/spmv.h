// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// SPDX-License-Identifier:    MIT

#include "L2GMap.h"
#include "Matrix.h"
#include "mpi_types.h"
#include "cg.h"
#ifdef USE_CUDA
#include "cuda/cg_cuda.h"
#endif
#ifdef _SYCL
#include "sycl/cg_sycl.h"
#endif
#include "read_petsc.h"

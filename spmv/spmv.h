// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// Copyright (C) 2021 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "config.h"

#include "L2GMap.h"
#include "Matrix.h"
#include "cg.h"
#ifdef _CUDA
#include "cuda/cg_cuda.h"
#endif
#ifdef _SYCL
#include "sycl/cg_sycl.h"
#endif
#include "read_petsc.h"

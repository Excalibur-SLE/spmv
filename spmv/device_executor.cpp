// Copyright (C) 2022 Athena Elafrou (ae488@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include "device_executor.h"

namespace spmv
{

void DeviceExecutor::free(void* ptr) const { _free(ptr); }

} // namespace spmv

#pragma once

// CUTLASS 3.4 gives every cp.async an L2::128B prefetch hint whenever CUDA >= 11.4 (cutlass/arch/memory.h, no
// switch). The encoder GEMM loads 64-byte K slices, and on sm_120 the hint doubles the L2 traffic of each load:
// Nsight Compute on the encoder's QKV product, same 8.1M LDGSTS, 236M L2 read sectors vs 119M for cuBLAS's
// kernel of the same configuration, L2 97% busy vs 64%, 4.24 vs 2.59 ms. Files compiled with this directory
// first on the include path (cuda/encoder_gemm.cu) get the header without the hint; the loads are the same.

#include "cutlass/arch/memory.h"
#undef CUTLASS_ENABLE_L2_PREFETCH
#define CUTLASS_ENABLE_L2_PREFETCH 0
#include "../../../../../third_party/cutlass/include/cutlass/arch/memory_sm80.h"

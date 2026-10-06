#pragma once

#include <cuda_fp16.h>

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    struct SharedMemoryRows;

    // CT2_LADDER_CROSS=1: the cross-attention products of a clip's sampled rows (shared_memory_rows.h: one query a
    // row and head against the clip's 1500 keys of 64 dims) in kernels that read the clip's keys and values once for
    // all its rows, each row with the arithmetic of the cuBLAS call its group makes (cuda/clip_groups.h: a group of
    // n rows is one call of n x 20 entries), recovered on the L40S with cuBLAS 12.9.2 by
    // tools/turing/kernels/ladder_cross_probe.cu (lcross2, every candidate against cuBLAS over 3 fills):
    //   scores, 20 or 40 entries (cuBLAS's gemv): 4 partials over the 64 dims, dim d in partial d % 4, combined
    //   (p0 + p2) + (p1 + p3); 60, 80 or 100 (a tensor-core kernel): an mma.sync m16n8k16 chain over the dims in
    //   groups of 16; then half(alpha * sum)
    //   output, 20 entries: 32 partials of 47 consecutive keys; 40, 60 or 80: 16 partials, key i in partial i % 16;
    //   100: 4 partials, key i in partial i % 4; each partial in key order, combined by a tree from the halves
    //   (s_r += s_{r + T/2}, ... s_r += s_{r + 1}); then half(sum)
    // Products of halves are exact in fp32, so a fused multiply-add is the product's sum. Only one clip a call
    // (a long recording's ladder) and groups of 1..5 rows; false (nothing launched) otherwise, or on another device.
    bool ladder_cross_scores(const SharedMemoryRows& rows, const __half* q, const __half* k, __half* scores,
                             dim_t heads, dim_t keys, dim_t depth, float alpha);
    bool ladder_cross_output(const SharedMemoryRows& rows, const __half* p, const __half* v, __half* out, dim_t heads,
                             dim_t keys, dim_t depth);

  }
}

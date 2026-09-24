#pragma once

// Decoder Dense layers at a few rows: C[m][n] = sum_k A[m][k] * W[n][k] (fp16, COMPUTE_32F, alpha 1,
// beta 0), with the exact arithmetic of the cuBLAS 12.9.2 kernels on sm_75, recovered by
// tools/turing/kernels/gemm_probe.cu. Every cuBLAS kernel here accumulates 8-wide k-groups with the
// tensor-core instruction mma.sync m16n8k8 (f32 accumulators from zero, groups in increasing order):
//   recipe 1: one chain over all of k; out = half(c)
//   recipe 2: two chains, 32-k blocks alternating (chain s takes blocks s, s + 2, ...); out = half(c0 + c1)
//   recipe 3: split-K in 4 contiguous quarters, each done as recipe 2 and rounded to half; the four
//             partials added forward in fp32 (((p0 + p1) + p2) + p3); out = half(sum)
// Same instruction, same groups, same order: the same bits, with one warp per (8 columns, chain) and
// no separate reduction kernel. Checked on every shape by gemm_check.cu. Not used by the library:
// the replica only matches cuBLAS's speed (a win needs a CUTLASS-grade kernel); kept with its proofs.

#include <cstdint>

namespace ctranslate2 {
  namespace cuda {

    // The recipe of the cuBLAS call at these sizes (the decoder's rows are 5 beams per batch entry),
    // or 0 when the shape was not verified.
    inline int small_m_gemm_recipe(int64_t m, int64_t n, int64_t k) {
      if (m < 5 || m > 40 || m % 5 != 0)
        return 0;
      const bool narrow = n == 1280 && (k == 1280 || k == 5120);
      const bool wide = (n == 3840 || n == 5120) && k == 1280;
      if (!narrow && !wide)
        return 0;
      if (m <= 15)
        return 1;
      return wide ? 2 : (m >= 35 ? 3 : 1);
    }

  }
}

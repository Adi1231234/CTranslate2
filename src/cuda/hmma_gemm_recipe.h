#pragma once

// Decoder Dense layers at a few rows: C[m][n] = sum_k A[m][k] * W[n][k] (fp16, COMPUTE_32F, alpha 1, beta 0),
// with the exact arithmetic of the cuBLAS 12.9.2 kernels on sm_120 (the store PC's RTX 5060 Ti), recovered by
// tools/turing/kernels/hmma_probe.cu (rows 1..48, four fills each, bit for bit). Every such kernel accumulates
// 16-wide k groups with the tensor-core instruction mma.sync m16n8k16 (f32 accumulators from zero, groups in
// increasing order):
//   recipe 1: one chain over all of k; out = half(c)
//   recipe 3: serial split-K in 3 slices of 1728 (ceil(k / 3) rounded up to 32), each a chain from zero;
//             out = half(c0), then out = half(c_s + out) for s = 1, 2
// Rows 1 and, at k = 5120, rows 17, 18, 23 and 24 run other kernels. Not used by the library: with the
// weights read from DRAM, hmma_gemm.cuh is exact on every shape but no faster than cuBLAS (hmma_check.cu),
// and in production it was slower (25.9 vs 24.2 s on the store PC's 150 clips); kept with its proofs.

#include <cstdint>

namespace ctranslate2 {
  namespace cuda {

    inline int hmma_gemm_recipe(int64_t m, int64_t n, int64_t k) {
      if (m < 2 || m > 48)
        return 0;
      if (k == 1280 && (n == 1280 || n == 3840 || n == 5120 || n == 51866))
        return 1;
      if (k == 5120 && n == 1280) {
        if (m <= 16)
          return 1;
        if ((m >= 19 && m <= 22) || m >= 25)
          return 3;
      }
      return 0;
    }

    inline int hmma_gemm_slices(int recipe) {
      return recipe == 3 ? 3 : 1;
    }

    inline int64_t hmma_gemm_slice_k(int64_t k, int slices) {
      return slices == 1 ? k : ((k + slices - 1) / slices + 31) / 32 * 32;
    }

    // Bytes of float partial results recipe 3 needs (0 for recipe 1).
    inline size_t hmma_gemm_workspace_bytes(int64_t m, int64_t n, int64_t k) {
      const int slices = hmma_gemm_slices(hmma_gemm_recipe(m, n, k));
      return slices == 1 ? 0 : sizeof (float) * slices * m * n;
    }

  }
}

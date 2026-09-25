#pragma once

// Phase 0 of cross_attention.cuh at the decoding steps: the block's queries projected from the normed decoder
// state, q[j][d] = __hadd(bias[64h + d], half(sum_k x[c m + j][k] W[64h + d][k])): the arithmetic of the Dense
// layer, cuBLAS 12.9.2 on sm_120 at 2..48 rows of 1280 x 1280 (one mma.sync m16n8k16 chain over k from zero with
// the activations as the A operand, hmma_gemm_recipe.h recipe 1; tools/turing/kernels/hmma_check.cu), and
// split_heads_bias's bias add (cuda/split_heads.cu). An output's arithmetic does not depend on the rows beside it.

#include "exact_attention_parts.cuh"

namespace at {
  namespace native {

    __device__ __forceinline__ unsigned ca_word(const __half* p) {
      return *reinterpret_cast<const unsigned*>(p);
    }

    // Rows j0 .. j0 + rows - 1 (rows <= 8) of clip `clip`'s m queries, head `head`, K inputs: x is [clips * m][K],
    // w [heads * 64][K], bias [heads * 64]. Warp w computes dims 16w .. 16w + 15 into qs[j - j0][d] (pitch halves).
    __device__ __forceinline__ void ca_project_queries(const __half* x, const __half* w, const __half* bias,
                                                       __half* qs, int pitch, int clip, int head, int m, int j0,
                                                       int rows, int K) {
      const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
      const __half* xr = x + (size_t)(clip * m + j0 + g) * K + 2 * t;      // A row g (zero past the rows)
      const __half* w0 = w + (size_t)(64 * head + 16 * warp + g) * K + 2 * t;  // B columns: dims 16w + g, + 8
      const __half* w1 = w0 + (size_t)8 * K;
      float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll 4
      for (int k0 = 0; k0 < K; k0 += 16) {                   // 16-wide k groups in increasing order
        const unsigned a0 = g < rows ? ca_word(xr + k0) : 0u, a2 = g < rows ? ca_word(xr + k0 + 8) : 0u;
        ea_mma(d0, a0, 0u, a2, 0u, make_uint2(ca_word(w0 + k0), ca_word(w0 + k0 + 8)));
        ea_mma(d1, a0, 0u, a2, 0u, make_uint2(ca_word(w1 + k0), ca_word(w1 + k0 + 8)));
      }
      if (g < rows)                                          // d: (row g, cols 2t, 2t + 1), rows g + 8 unused
        #pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
          const float* d = tile ? d1 : d0;
          const int dim = 16 * warp + 8 * tile + 2 * t;
          const __half2 b = *reinterpret_cast<const __half2*>(bias + 64 * head + dim);
          *reinterpret_cast<__half2*>(qs + g * pitch + dim) = __hadd2(b, __floats2half2_rn(d[0], d[1]));
        }
    }

  }
}

#pragma once

// ladder_cross.cuh's scores kernels (a ladder's rows' queries against their clip's memory keys).

#include "cuda/ladder_cross_common.cuh"

namespace ctranslate2 {
  namespace cuda {

    // Scores of rows whose groups are 1 or 2 rows (cuBLAS's gemv): a thread per (key, row), blockIdx.z the row.
    __global__ void lc_scores_gemv(const __half* q, const __half* k, __half* scores, LadderRows rows, int heads,
                                   float alpha) {
      const int i = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y, r = rows.row[blockIdx.z];
      if (i >= lc_keys)
        return;
      const uint4* kv = reinterpret_cast<const uint4*>(k + (static_cast<size_t>(h) * lc_keys + i) * lc_depth);
      const uint4* qv = reinterpret_cast<const uint4*>(q + (static_cast<size_t>(r) * heads + h) * lc_depth);
      float part[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int v = 0; v < lc_depth / 8; ++v) {               // dims 8v .. 8v + 7 in order
        const uint4 a = __ldg(kv + v), b = __ldg(qv + v);
        const __half* ah = reinterpret_cast<const __half*>(&a);
        const __half* bh = reinterpret_cast<const __half*>(&b);
        #pragma unroll
        for (int u = 0; u < 8; ++u)
          part[u & 3] = fmaf(hf(ah[u]), hf(bh[u]), part[u & 3]);
      }
      part[0] += part[2];
      part[1] += part[3];
      part[0] += part[1];
      scores[(static_cast<size_t>(r) * heads + h) * lc_keys + i] = __float2half_rn(alpha * part[0]);
    }

    // Scores of rows whose groups are 3..5 rows (cuBLAS's tensor-core kernel): a warp per (16 keys, head), the keys
    // the rows of A and 8 rows' queries the columns of B, an mma.sync m16n8k16 chain over the dims.
    __global__ void lc_scores_mma(const __half* q, const __half* k, __half* scores, LadderRows rows, int heads,
                                  float alpha) {
      const int lane = threadIdx.x, g = lane >> 2, t = lane & 3, i0 = blockIdx.x * 16, h = blockIdx.y;
      const __half* kb = k + static_cast<size_t>(h) * lc_keys * lc_depth;
      const auto key_pair = [&](int i, int d) { return i < lc_keys ? pair(kb + static_cast<size_t>(i) * lc_depth + d)
                                                                   : 0u; };
      unsigned a[4][4];
      #pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int d0 = 16 * s;
        a[s][0] = key_pair(i0 + g, d0 + 2 * t);
        a[s][1] = key_pair(i0 + g + 8, d0 + 2 * t);
        a[s][2] = key_pair(i0 + g, d0 + 8 + 2 * t);
        a[s][3] = key_pair(i0 + g + 8, d0 + 8 + 2 * t);
      }
      for (int c0 = 0; c0 < rows.count; c0 += 8) {
        const int col = c0 + g;
        const __half* qc = col < rows.count
          ? q + (static_cast<size_t>(rows.row[col]) * heads + h) * lc_depth : nullptr;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
          const int d0 = 16 * s;
          const unsigned b0 = qc ? pair(qc + d0 + 2 * t) : 0u, b1 = qc ? pair(qc + d0 + 8 + 2 * t) : 0u;
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                       "{%0,%1,%2,%3};"
                       : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                       : "r"(a[s][0]), "r"(a[s][1]), "r"(a[s][2]), "r"(a[s][3]), "r"(b0), "r"(b1));
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {                        // C[key g (+8)][column 2t (+1)]
          const int i = i0 + g + 8 * (e / 2), c = c0 + 2 * t + e % 2;
          if (i < lc_keys && c < rows.count)
            scores[(static_cast<size_t>(rows.row[c]) * heads + h) * lc_keys + i] = __float2half_rn(alpha * acc[e]);
        }
      }
    }

  }
}

#pragma once

// Attention scores C[b][j][i] = alpha * dot(K[b][i][0:64], Q[b][j][0:64]) in fp16 with the exact fp32
// arithmetic of the cuBLAS kernel that cublasGemmStridedBatchedEx(COMPUTE_32F) runs for this shape on
// sm_75 (gemmSN_TN), recovered by tools/turing/kernels/qk_probe.cu:
//   p_r = k[r]q[r] + k[r+16]q[r+16] + k[r+32]q[r+32] + k[r+48]q[r+48]   (left to right, r = 0..15)
//   sum = p_0 + p_1 + ... + p_15                                        (left to right)
//   C   = half_rn(alpha * sum)
// Products of halves are exact in fp32, so contraction into FMA cannot change a result.
// cuBLAS spreads this over 8 keys per block; here one thread owns a key and reads its row once for
// all m queries, which keeps the loads streaming.

#include <cstdint>
#include <cuda_fp16.h>

namespace ctranslate2 {
  namespace cuda {

    constexpr int attention_scores_k64_keys_per_block = 128;
    constexpr int attention_scores_k64_max_queries = 8;

    __global__ void __launch_bounds__(attention_scores_k64_keys_per_block)
    attention_scores_k64_kernel(const __half* __restrict__ q,
                                const __half* __restrict__ k,
                                __half* __restrict__ c,
                                int m,
                                int n,
                                float alpha) {
      __shared__ float qs[attention_scores_k64_max_queries * 64];
      const int b = blockIdx.y;
      const __half* qb = q + static_cast<size_t>(b) * m * 64;
      for (int t = threadIdx.x; t < m * 64; t += blockDim.x)
        qs[t] = __half2float(qb[t]);
      __syncthreads();

      const int i = blockIdx.x * attention_scores_k64_keys_per_block + threadIdx.x;
      if (i >= n)
        return;
      const uint4* row = reinterpret_cast<const uint4*>(k + (static_cast<size_t>(b) * n + i) * 64);
      float kv[64];
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const uint4 w = __ldg(row + v);
        const __half2* h = reinterpret_cast<const __half2*>(&w);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __half22float2(h[e]);
          kv[v * 8 + 2 * e] = f.x;
          kv[v * 8 + 2 * e + 1] = f.y;
        }
      }

      __half* cb = c + static_cast<size_t>(b) * m * n + i;
#pragma unroll
      for (int j = 0; j < attention_scores_k64_max_queries; ++j) {
        if (j >= m)
          break;
        const float* qj = qs + j * 64;
        float sum = 0;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
          float p = kv[r] * qj[r];
          p = fmaf(kv[r + 16], qj[r + 16], p);
          p = fmaf(kv[r + 32], qj[r + 32], p);
          p = fmaf(kv[r + 48], qj[r + 48], p);
          sum = r == 0 ? p : sum + p;
        }
        cb[static_cast<size_t>(j) * n] = __float2half_rn(alpha * sum);
      }
    }

    // Q [batch, m, 64], K [batch, n, 64], C [batch, m, n], all contiguous.
    inline void attention_scores_k64(const __half* q, const __half* k, __half* c,
                                     int batch, int m, int n, float alpha, cudaStream_t stream) {
      const dim3 grid((n + attention_scores_k64_keys_per_block - 1) / attention_scores_k64_keys_per_block,
                      batch);
      attention_scores_k64_kernel<<<grid, attention_scores_k64_keys_per_block, 0, stream>>>(
        q, k, c, m, n, alpha);
    }

  }
}

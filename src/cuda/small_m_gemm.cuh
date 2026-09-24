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
// no separate reduction kernel. Checked on every routed shape by tools/turing/kernels/gemm_check.cu.

#include <cstdint>
#include <cuda_fp16.h>

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

    __device__ __forceinline__ void smg_mma(float* d, unsigned a0, unsigned a1, unsigned b) {
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b));
    }

    __device__ __forceinline__ unsigned smg_ld(const __half* p) {
      return __ldg(reinterpret_cast<const unsigned*>(p));
    }

    // Warps per block, and warps per 8-column tile (one per chain).
    template <int Recipe> struct smg_shape {
      static constexpr int warps = Recipe == 3 ? 8 : 4;
      static constexpr int per_tile = Recipe == 1 ? 1 : Recipe == 2 ? 2 : 8;
    };

    template <int Recipe, int MT>
    __global__ void __launch_bounds__(smg_shape<Recipe>::warps * 32)
    small_m_gemm_kernel(const __half* A, const __half* W, __half* C, int M, int N, int K) {
      using S = smg_shape<Recipe>;
      __shared__ float part[S::warps][MT * 4][32];
      const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g4 = lane / 4, t4 = lane % 4;
      const int role = warp % S::per_tile;                     // recipe 2: chain; 3: 2 * quarter + chain
      const int n0 = (blockIdx.x * (S::warps / S::per_tile) + warp / S::per_tile) * 8;
      const int blocks = K / 32, first = Recipe == 3 ? (role / 2) * blocks / 4 : 0;
      const int last = Recipe == 3 ? first + blocks / 4 : blocks, chain = Recipe == 1 ? 0 : role % 2;
      const __half* w = W + (size_t)(n0 + g4) * K + 2 * t4;
      float acc[MT][4] = {};
      for (int blk = first + chain * (Recipe != 1); blk < last; blk += Recipe == 1 ? 1 : 2) {
        const int k0 = blk * 32;
        unsigned b[4], a[MT][4][2];
        #pragma unroll
        for (int j = 0; j < 4; ++j) b[j] = smg_ld(w + k0 + 8 * j);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const int r0 = mt * 16 + g4, r1 = r0 + 8;
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            a[mt][j][0] = r0 < M ? smg_ld(A + (size_t)r0 * K + k0 + 8 * j + 2 * t4) : 0u;
            a[mt][j][1] = r1 < M ? smg_ld(A + (size_t)r1 * K + k0 + 8 * j + 2 * t4) : 0u;
          }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j)
          #pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            smg_mma(acc[mt], a[mt][j][0], a[mt][j][1], b[j]);
      }
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) part[warp][mt * 4 + e][lane] = acc[mt][e];
      __syncthreads();
      if (role != 0)
        return;
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = mt * 4 + e;
          float total = part[warp][i][lane];
          if (Recipe == 2)
            total = total + part[warp + 1][i][lane];
          if (Recipe == 3)
            for (int q = 0; q < 4; ++q) {
              const float p = __half2float(__float2half_rn(part[warp + 2 * q][i][lane] + part[warp + 2 * q + 1][i][lane]));
              total = q == 0 ? p : total + p;
            }
          const int row = mt * 16 + g4 + 8 * (e / 2), col = n0 + 2 * t4 + e % 2;
          if (row < M)
            C[(size_t)row * N + col] = __float2half_rn(total);
        }
    }

    template <int Recipe>
    void small_m_gemm_launch(const __half* A, const __half* W, __half* C, int M, int N, int K,
                             cudaStream_t stream) {
      using S = smg_shape<Recipe>;
      const int grid = N / 8 / (S::warps / S::per_tile), threads = S::warps * 32;
      if (M <= 16)
        small_m_gemm_kernel<Recipe, 1><<<grid, threads, 0, stream>>>(A, W, C, M, N, K);
      else if (M <= 32)
        small_m_gemm_kernel<Recipe, 2><<<grid, threads, 0, stream>>>(A, W, C, M, N, K);
      else
        small_m_gemm_kernel<Recipe, 3><<<grid, threads, 0, stream>>>(A, W, C, M, N, K);
    }

    // C = A W^T for a verified shape (small_m_gemm_recipe(M, N, K) != 0). A, W and C 4-byte aligned.
    inline void small_m_gemm(const __half* A, const __half* W, __half* C, int M, int N, int K,
                             cudaStream_t stream) {
      switch (small_m_gemm_recipe(M, N, K)) {
      case 1: small_m_gemm_launch<1>(A, W, C, M, N, K, stream); break;
      case 2: small_m_gemm_launch<2>(A, W, C, M, N, K, stream); break;
      case 3: small_m_gemm_launch<3>(A, W, C, M, N, K, stream); break;
      default: break;
      }
    }

  }
}

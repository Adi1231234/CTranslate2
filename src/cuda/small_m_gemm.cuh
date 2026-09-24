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

    __device__ __forceinline__ unsigned smg_lds(const __half* p) {
      return *reinterpret_cast<const unsigned*>(p);
    }

    // Warps are (n-tile, role): a role is one chain (recipe 2) or one (quarter, chain) (recipe 3).
    // All warps walk k in steps of 64 per quarter, with the block's rows of A for the step staged in
    // shared memory (double-buffered, 72-half rows: conflict-free fragment reads) and each warp's
    // weights for the next step loaded ahead into registers.
    template <int Recipe> struct smg_cfg {
      static constexpr int roles = Recipe == 1 ? 1 : Recipe == 2 ? 2 : 8;
      static constexpr int quarters = Recipe == 3 ? 4 : 1;
      static constexpr int tiles = Recipe == 3 ? 1 : 4;              // 8-column tiles per block
      static constexpr int warps = roles * tiles;
      static constexpr int groups = Recipe == 1 ? 8 : 4;             // k-groups per warp and step
    };
    constexpr int smg_row = 72;

    template <int Recipe, int MT>
    constexpr size_t smg_smem_bytes() {
      using S = smg_cfg<Recipe>;
      const size_t a = 2 * S::quarters * MT * 16 * smg_row * sizeof (__half);
      const size_t part = S::warps * MT * 4 * 32 * sizeof (float);
      return a > part ? a : part;
    }

    template <int Recipe, int MT>
    __global__ void __launch_bounds__(smg_cfg<Recipe>::warps * 32)
    small_m_gemm_kernel(const __half* A, const __half* W, __half* C, int M, int N, int K) {
      using S = smg_cfg<Recipe>;
      extern __shared__ __align__(16) unsigned char smg_smem[];
      __half* a_s = reinterpret_cast<__half*>(smg_smem);
      constexpr int rows = MT * 16, buf = S::quarters * rows * smg_row;
      const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g4 = lane / 4, t4 = lane % 4;
      const int role = warp % S::roles, n0 = (blockIdx.x * S::tiles + warp / S::roles) * 8;
      const int quarter = Recipe == 3 ? role / 2 : 0, chain = Recipe == 1 ? 0 : role % 2;
      const int range = K / S::quarters, steps = range / 64, shift = Recipe == 1 ? 0 : chain * 32;
      const __half* w = W + (size_t)(n0 + g4) * K + quarter * range + shift + 2 * t4;
      auto stage = [&](int s, int b) {
        for (int v = threadIdx.x; v < S::quarters * rows * 8; v += blockDim.x) {
          const int q = v / (rows * 8), r = (v / 8) % rows, c = v % 8;
          uint4 x = make_uint4(0, 0, 0, 0);
          if (r < M)
            x = __ldg(reinterpret_cast<const uint4*>(A + (size_t)r * K + q * range + s * 64 + c * 8));
          *reinterpret_cast<uint4*>(a_s + b * buf + (q * rows + r) * smg_row + c * 8) = x;
        }
      };
      unsigned wb[S::groups], wn[S::groups];
      auto load_w = [&](int s, unsigned* b) {
        #pragma unroll
        for (int j = 0; j < S::groups; ++j)
          b[j] = __ldg(reinterpret_cast<const unsigned*>(w + s * 64 + 8 * j));
      };
      float acc[MT][4] = {};
      stage(0, 0);
      load_w(0, wb);
      __syncthreads();
      for (int s = 0; s < steps; ++s) {
        if (s + 1 < steps) {
          stage(s + 1, (s + 1) & 1);
          load_w(s + 1, wn);
        }
        const __half* a = a_s + (s & 1) * buf + quarter * rows * smg_row + shift + 2 * t4;
        #pragma unroll
        for (int j = 0; j < S::groups; ++j)            // this chain's groups, in increasing k
          #pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            const __half* p = a + (mt * 16 + g4) * smg_row + 8 * j;
            smg_mma(acc[mt], smg_lds(p), smg_lds(p + 8 * smg_row), wb[j]);
          }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < S::groups; ++j) wb[j] = wn[j];
      }
      float* part = reinterpret_cast<float*>(smg_smem);          // [warps][MT * 4][32], A is done
      #pragma unroll
      for (int i = 0; i < MT * 4; ++i) part[(warp * MT * 4 + i) * 32 + lane] = acc[i / 4][i % 4];
      __syncthreads();
      if (role != 0)
        return;
      #pragma unroll
      for (int i = 0; i < MT * 4; ++i) {
        auto at = [&](int wp) { return part[((warp + wp) * MT * 4 + i) * 32 + lane]; };
        float total = at(0);
        if (Recipe == 2)
          total = total + at(1);
        if (Recipe == 3)
          for (int q = 0; q < 4; ++q) {
            const float p = __half2float(__float2half_rn(at(2 * q) + at(2 * q + 1)));
            total = q == 0 ? p : total + p;
          }
        const int row = (i / 4) * 16 + g4 + 8 * ((i % 4) / 2), col = n0 + 2 * t4 + i % 2;
        if (row < M)
          C[(size_t)row * N + col] = __float2half_rn(total);
      }
    }

    template <int Recipe, int MT>
    void small_m_gemm_run(const __half* A, const __half* W, __half* C, int M, int N, int K,
                          cudaStream_t stream) {
      using S = smg_cfg<Recipe>;
      constexpr size_t smem = smg_smem_bytes<Recipe, MT>();
      static const bool configured = [] {           // opt in above the 48 KB default once
        return cudaFuncSetAttribute(small_m_gemm_kernel<Recipe, MT>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)) == cudaSuccess;
      }();
      (void)configured;
      small_m_gemm_kernel<Recipe, MT><<<N / 8 / S::tiles, S::warps * 32, smem, stream>>>(A, W, C, M, N, K);
    }

    template <int Recipe>
    void small_m_gemm_launch(const __half* A, const __half* W, __half* C, int M, int N, int K,
                             cudaStream_t stream) {
      if (M <= 16)
        small_m_gemm_run<Recipe, 1>(A, W, C, M, N, K, stream);
      else if (M <= 32)
        small_m_gemm_run<Recipe, 2>(A, W, C, M, N, K, stream);
      else
        small_m_gemm_run<Recipe, 3>(A, W, C, M, N, K, stream);
    }

    // C = A W^T for a verified shape (small_m_gemm_recipe(M, N, K) != 0). A 16-byte aligned, W and C 4.
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

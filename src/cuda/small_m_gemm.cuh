#pragma once

// The kernel of small_m_gemm_recipe.h. A block owns 8 * tiles columns and walks k in steps of 64 per
// split-K quarter: each step's weights and rows of A go to shared memory with 16-byte loads (the next
// step's already in registers), warps take their fragments with ldmatrix and run their chain's
// mma.sync steps. A warp is one chain (and quarter) over two 8-column tiles; the chains of a column
// are combined in the block, in the recipe's order, so there is no separate reduction kernel.

#include <cstdint>
#include <cuda_fp16.h>

#include "small_m_gemm_recipe.h"

namespace ctranslate2 {
  namespace cuda {

    __device__ __forceinline__ void smg_mma(float* d, unsigned a0, unsigned a1, unsigned b) {
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b));
    }

    __device__ __forceinline__ void smg_ldm4(unsigned* r, const __half* p) {
      const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    }

    template <int Recipe> struct smg_cfg {
      static constexpr int chains = Recipe == 1 ? 1 : 2, quarters = Recipe == 3 ? 4 : 1;
      static constexpr int tiles = Recipe == 3 ? 2 : 4;                 // 8-column tiles per block
      static constexpr int warps = chains * quarters * tiles / 2;       // two tiles per warp
      static constexpr int kw = Recipe == 1 ? 64 : 32;                  // k per warp and step
    };
    constexpr int smg_row = 72;                                         // halves per staged row

    template <int Recipe, int MT> struct smg_stage {                    // one step in shared memory
      static constexpr int rows = MT * 16 + smg_cfg<Recipe>::tiles * 8;  // rows of A, then weights
      static constexpr int halves = smg_cfg<Recipe>::quarters * rows * smg_row;
      static constexpr int vecs = smg_cfg<Recipe>::quarters * rows * 8;  // 16-byte global loads
    };

    template <int Recipe, int MT>
    __global__ void __launch_bounds__(smg_cfg<Recipe>::warps * 32)
    small_m_gemm_kernel(const __half* A, const __half* W, __half* C, int M, int N, int K) {
      using S = smg_cfg<Recipe>;
      using G = smg_stage<Recipe, MT>;
      constexpr int threads = S::warps * 32, per_thread = (G::vecs + threads - 1) / threads;
      extern __shared__ __align__(16) unsigned char smg_smem[];
      __half* st = reinterpret_cast<__half*>(smg_smem);
      const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, lr = lane % 8, lm = lane / 8;
      const int pair = warp % (S::tiles / 2), role = warp / (S::tiles / 2);
      const int quarter = role / S::chains, chain = role % S::chains;
      const int n0 = blockIdx.x * S::tiles * 8, range = K / S::quarters, steps = range / 64;
      uint4 pre[per_thread];
      auto fetch = [&](int s) {                             // global -> registers
        #pragma unroll
        for (int i = 0; i < per_thread; ++i) {
          const int v = threadIdx.x + i * threads;
          const int q = v / (G::rows * 8), r = (v / 8) % G::rows, k = q * range + s * 64 + (v % 8) * 8;
          const __half* src = r < MT * 16 ? (r < M ? A + (size_t)r * K + k : nullptr)
                                          : W + (size_t)(n0 + r - MT * 16) * K + k;
          pre[i] = v < G::vecs && src ? __ldg(reinterpret_cast<const uint4*>(src)) : make_uint4(0, 0, 0, 0);
        }
      };
      float acc[2][MT][4] = {};
      fetch(0);
      for (int s = 0; s < steps; ++s) {
        #pragma unroll
        for (int i = 0; i < per_thread; ++i) {              // registers -> shared memory
          const int v = threadIdx.x + i * threads;
          if (v < G::vecs)
            *reinterpret_cast<uint4*>(st + (v / 8) * smg_row + (v % 8) * 8) = pre[i];
        }
        __syncthreads();
        if (s + 1 < steps)
          fetch(s + 1);
        const __half* base = st + quarter * G::rows * smg_row + (S::chains == 2 ? chain * 32 : 0);
        #pragma unroll
        for (int h = 0; h < S::kw / 32; ++h) {             // this chain's 32-k blocks, in order
          unsigned b[2][4];                                 // 4 groups of each of the two tiles
          #pragma unroll
          for (int t = 0; t < 2; ++t)
            smg_ldm4(b[t], base + (MT * 16 + (2 * pair + t) * 8 + lr) * smg_row + h * 32 + 8 * lm);
          #pragma unroll
          for (int g2 = 0; g2 < 2; ++g2)                    // groups 2 * g2 and 2 * g2 + 1
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              unsigned a[4];
              smg_ldm4(a, base + (mt * 16 + (lm % 2) * 8 + lr) * smg_row + h * 32 + 16 * g2 + 8 * (lm / 2));
              #pragma unroll
              for (int t = 0; t < 2; ++t) {
                smg_mma(acc[t][mt], a[0], a[1], b[t][2 * g2]);
                smg_mma(acc[t][mt], a[2], a[3], b[t][2 * g2 + 1]);
              }
            }
        }
        __syncthreads();
      }
      float* part = reinterpret_cast<float*>(smg_smem);    // [role][tile][MT * 4][32]
      constexpr int per_role = S::tiles * MT * 4 * 32;
      #pragma unroll
      for (int t = 0; t < 2; ++t)
        #pragma unroll
        for (int i = 0; i < MT * 4; ++i)
          part[role * per_role + ((2 * pair + t) * MT * 4 + i) * 32 + lane] = acc[t][i / 4][i % 4];
      __syncthreads();
      for (int o = threadIdx.x; o < per_role; o += threads) {
        const int tile = o / (MT * 4 * 32), i = (o / 32) % (MT * 4), l = o % 32;
        auto at = [&](int r) { return part[r * per_role + o]; };
        float total = at(0);
        if (Recipe == 2)
          total = total + at(1);
        if (Recipe == 3)
          for (int q = 0; q < 4; ++q) {
            const float p = __half2float(__float2half_rn(at(2 * q) + at(2 * q + 1)));
            total = q == 0 ? p : total + p;
          }
        const int row = (i / 4) * 16 + l / 4 + 8 * ((i % 4) / 2), col = n0 + tile * 8 + 2 * (l % 4) + i % 2;
        if (row < M)
          C[(size_t)row * N + col] = __float2half_rn(total);
      }
    }

    template <int Recipe, int MT>
    constexpr size_t smg_smem_bytes() {
      using S = smg_cfg<Recipe>;
      const size_t stage = smg_stage<Recipe, MT>::halves * sizeof (__half);
      const size_t part = S::chains * S::quarters * S::tiles * MT * 4 * 32 * sizeof (float);
      return stage > part ? stage : part;
    }

    template <int Recipe, int MT>
    void small_m_gemm_run(const __half* A, const __half* W, __half* C, int M, int N, int K,
                          cudaStream_t stream) {
      static const bool configured = [] {                  // opt in above the 48 KB default once
        return cudaFuncSetAttribute(small_m_gemm_kernel<Recipe, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(smg_smem_bytes<Recipe, MT>())) == cudaSuccess;
      }();
      (void)configured;
      small_m_gemm_kernel<Recipe, MT><<<N / 8 / smg_cfg<Recipe>::tiles, smg_cfg<Recipe>::warps * 32,
                                        smg_smem_bytes<Recipe, MT>(), stream>>>(A, W, C, M, N, K);
    }

    template <int Recipe>
    void small_m_gemm_rows(const __half* A, const __half* W, __half* C, int M, int N, int K,
                           cudaStream_t stream) {
      if (M <= 16)
        small_m_gemm_run<Recipe, 1>(A, W, C, M, N, K, stream);
      else if (M <= 32)
        small_m_gemm_run<Recipe, 2>(A, W, C, M, N, K, stream);
      else
        small_m_gemm_run<Recipe, 3>(A, W, C, M, N, K, stream);
    }

    // C = A W^T for a verified shape (small_m_gemm_recipe(M, N, K) != 0); 16-byte aligned rows.
    inline void small_m_gemm(const __half* A, const __half* W, __half* C, int M, int N, int K,
                             cudaStream_t stream) {
      switch (small_m_gemm_recipe(M, N, K)) {
      case 1: small_m_gemm_rows<1>(A, W, C, M, N, K, stream); break;
      case 2: small_m_gemm_rows<2>(A, W, C, M, N, K, stream); break;
      case 3: small_m_gemm_rows<3>(A, W, C, M, N, K, stream); break;
      default: break;
      }
    }

  }
}

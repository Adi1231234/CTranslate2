#pragma once

// The kernel of hmma_gemm_recipe.h (launched by hmma_gemm.cuh). cuBLAS runs these few-row products as
// 16..64-wide tiles on few blocks and reads the weights at 20..75% of the memory bandwidth. Here a warp owns 16 columns (two n8 tiles for
// every 16-row tile of A) and streams their weights once, 64 k at a time, through a 4-stage cp.async
// pipeline; fragments come from shared memory with ldmatrix in the standard mma layout, and the chain
// runs the same mma.sync m16n8k16 steps in the same order, so the results are the same bits. Recipe 3's
// slices run as separate blocks (blockIdx.y) writing float partials, combined in slice order afterwards.

#include <cstdint>
#include <cuda_fp16.h>

#include "hmma_gemm_recipe.h"

namespace ctranslate2 {
  namespace cuda {

    constexpr int hg_stages = 4, hg_kstep = 64, hg_pitch = 72;          // halves per staged row

    __device__ __forceinline__ void hg_ldm4(unsigned* r, const __half* p) {
      const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    }

    __device__ __forceinline__ void hg_cp16(__half* dst, const __half* src, bool valid) {
#if __CUDA_ARCH__ >= 800
      const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(valid ? 16 : 0));
#endif
    }

    __device__ __forceinline__ void hg_mma(float* d, const unsigned* a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
    }

    // Block: Warps x 16 columns of one k slice (blockIdx.y, slice_k long); out to C, or to P when sliced.
    template <int MT, int Warps>
    __global__ void __launch_bounds__(Warps * 32)
    hmma_gemm_kernel(const __half* A, const __half* W, __half* C, float* P, int M, int N, int K, int slice_k) {
#if __CUDA_ARCH__ >= 800                                     // cp.async and m16n8k16: sm_80 and newer
      extern __shared__ __align__(16) unsigned char hg_smem[];
      __half* st = reinterpret_cast<__half*>(hg_smem);
      constexpr int rows = MT * 16 + Warps * 16, stage = rows * hg_pitch;
      const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, lr = lane % 8, lm = lane / 8;
      const int n0 = blockIdx.x * Warps * 16, k0 = blockIdx.y * slice_k;
      const int steps = (min(K, k0 + slice_k) - k0) / hg_kstep;
      auto load = [&](int s) {                               // one 64-k step of A's rows and the weights
        __half* dst = st + (s % hg_stages) * stage;
        for (int v = threadIdx.x; v < rows * 8; v += Warps * 32) {
          const int r = v / 8, c = (v % 8) * 8;
          const bool is_a = r < MT * 16;
          const int row = is_a ? r : n0 + r - MT * 16;
          const bool valid = row < (is_a ? M : N);          // missing rows load as zeros
          hg_cp16(dst + r * hg_pitch + c, (is_a ? A : W) + (size_t)(valid ? row : 0) * K + k0 + s * hg_kstep + c, valid);
        }
      };
      for (int s = 0; s < hg_stages - 1; ++s) {
        if (s < steps)
          load(s);
        asm volatile("cp.async.commit_group;");
      }
      float acc[2][MT][4] = {};
      for (int s = 0; s < steps; ++s) {
        asm volatile("cp.async.wait_group %0;" :: "n"(hg_stages - 2));
        __syncthreads();                                    // step s landed; step s - 1's buffer is free
        if (s + hg_stages - 1 < steps)
          load(s + hg_stages - 1);
        asm volatile("cp.async.commit_group;");
        const __half* base = st + (s % hg_stages) * stage;
        #pragma unroll
        for (int g = 0; g < hg_kstep / 16; ++g) {             // 16-k groups in increasing order
          unsigned b[4];                                      // tile 0: b0 b1, tile 1: b2 b3
          hg_ldm4(b, base + (MT * 16 + warp * 16 + (lm / 2) * 8 + lr) * hg_pitch + g * 16 + (lm % 2) * 8);
          #pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            unsigned a[4];
            hg_ldm4(a, base + (mt * 16 + (lm % 2) * 8 + lr) * hg_pitch + g * 16 + (lm / 2) * 8);
            hg_mma(acc[0][mt], a, b[0], b[1]);
            hg_mma(acc[1][mt], a, b[2], b[3]);
          }
        }
      }
      const int g = lane / 4, t = lane % 4;
      #pragma unroll
      for (int j = 0; j < 2; ++j)
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          #pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = mt * 16 + g + 8 * h, col = n0 + warp * 16 + j * 8 + 2 * t;
            if (row >= M || col >= N)
              continue;
            const float x = acc[j][mt][2 * h], y = acc[j][mt][2 * h + 1];
            if (P)
              *reinterpret_cast<float2*>(P + ((size_t)blockIdx.y * M + row) * N + col) = make_float2(x, y);
            else
              *reinterpret_cast<__half2*>(C + (size_t)row * N + col) = __floats2half2_rn(x, y);
          }
#endif
    }

    // Recipe 3's epilogue: out = half(p0), then out = half(p_s + out), slice by slice.
    static __global__ void hmma_gemm_combine(const float* P, __half* C, size_t count, int slices) {
      for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        float out = __half2float(__float2half_rn(P[i]));
        for (int s = 1; s < slices; ++s)
          out = __half2float(__float2half_rn(P[s * count + i] + out));
        C[i] = __float2half_rn(out);
      }
    }

  }
}

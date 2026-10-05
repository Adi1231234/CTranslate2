#pragma once

// C = A W^T (fp16, f32 accumulation) for the rows of several batches decoded together (cuda/clip_groups.h), each
// group of rows with the arithmetic cuBLAS gives a call with that group alone: split-K in `slices` slices of
// `slice` k, each slice one mma.sync m16n8k16 chain over its k in increasing 16-groups, rounded to half, the
// slices summed in order in fp32, out = half(sum) (grouped_split_gemm.cuh's arithmetic; one slice of all of k is
// cuBLAS's plain chain). Templated on the block's output tile TM x TN (4 warps, WARPS_M x 4 / WARPS_M), so the
// grid can have enough blocks for few rows: the arithmetic of an output does not depend on the tile.
// A [M x K], W [N x K], C [M x N] row-major; K a multiple of 32.

#include <cuda_fp16.h>

#include "cuda/grouped_split_gemm.cuh"

namespace ctranslate2 {
  namespace cuda {

    constexpr int tsg_kstep = 32, tsg_pitch = tsg_kstep + 8, tsg_stages = 4;   // halves a staged row

    template <int TM, int TN, int WARPS_M = 2>
    __global__ void __launch_bounds__(128)
    tiled_split_gemm_kernel(const __half* A, const __half* W, __half* C, int M, int N, int K, SplitGroups groups) {
#if __CUDA_ARCH__ >= 800
      constexpr int WARPS_N = 4 / WARPS_M, WM = TM / WARPS_M, WN = TN / WARPS_N, MT = WM / 16, NT = WN / 8;
      static_assert(MT >= 1 && NT >= 2 && NT % 2 == 0, "warp tile: 16 rows x 16 columns at least");
      extern __shared__ __align__(16) unsigned char tsg_smem[];
      __half* st = reinterpret_cast<__half*>(tsg_smem);
      constexpr int stage = (TM + TN) * tsg_pitch;           // A rows, then W rows
      const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, lr = lane % 8, lm = lane / 8;
      const int wm = warp / WARPS_N, wn = warp % WARPS_N;
      const int m0 = blockIdx.y * TM, n0 = blockIdx.x * TN, steps = K / tsg_kstep;
      const int g = lane / 4, t = lane % 4;
      int slice[MT][2], left[MT][2];                         // this thread's rows: slice length, k to its end
      bool pad[MT][2];
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = m0 + wm * WM + mt * 16 + g + 8 * h;
          const int grp = row < M ? gsg_group_of(groups, row) : -1;
          slice[mt][h] = grp < 0 ? 0 : groups.slice[grp];
          left[mt][h] = slice[mt][h];
          pad[mt][h] = grp >= 0 && groups.slices[grp] > (K + groups.slice[grp] - 1) / groups.slice[grp];
        }
      auto load = [&](int s) {
        __half* dst = st + (s % tsg_stages) * stage;
        for (int v = threadIdx.x; v < (TM + TN) * (tsg_kstep / 8); v += 128) {
          const int r = v / (tsg_kstep / 8), c = (v % (tsg_kstep / 8)) * 8;
          const bool is_a = r < TM;
          const int row = is_a ? m0 + r : n0 + r - TM;
          const bool valid = row < (is_a ? M : N);
          gsg_cp16(dst + r * tsg_pitch + c, (is_a ? A : W) + (size_t)(valid ? row : 0) * K + s * tsg_kstep + c,
                   valid);
        }
      };
      for (int s = 0; s < tsg_stages - 1; ++s) {
        if (s < steps)
          load(s);
        asm volatile("cp.async.commit_group;");
      }
      float acc[MT][NT][4] = {}, sum[MT][NT][4];
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
          #pragma unroll
          for (int e = 0; e < 4; ++e)
            sum[mt][nt][e] = -0.f;
      for (int s = 0; s < steps; ++s) {
        asm volatile("cp.async.wait_group %0;" :: "n"(tsg_stages - 2));
        __syncthreads();
        if (s + tsg_stages - 1 < steps)
          load(s + tsg_stages - 1);
        asm volatile("cp.async.commit_group;");
        const __half* base = st + (s % tsg_stages) * stage;
        #pragma unroll
        for (int q = 0; q < tsg_kstep / 16; ++q) {
          unsigned a[MT][4], b[NT / 2][4];                   // b[p]: n8 tiles 2p (b0 b1) and 2p + 1 (b2 b3)
          #pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            gsg_ldm4(a[mt], base + (wm * WM + mt * 16 + (lm % 2) * 8 + lr) * tsg_pitch + q * 16 + (lm / 2) * 8);
          #pragma unroll
          for (int p = 0; p < NT / 2; ++p)
            gsg_ldm4(b[p], base + (TM + wn * WN + p * 16 + (lm / 2) * 8 + lr) * tsg_pitch + q * 16 + (lm % 2) * 8);
          #pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (int p = 0; p < NT / 2; ++p) {
              gsg_mma(acc[mt][2 * p], a[mt], b[p][0], b[p][1]);
              gsg_mma(acc[mt][2 * p + 1], a[mt], b[p][2], b[p][3]);
            }
        }
        const bool last = s + 1 == steps;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          #pragma unroll
          for (int h = 0; h < 2; ++h) {
            left[mt][h] -= tsg_kstep;
            if (slice[mt][h] > 0 && (left[mt][h] == 0 || last)) {   // this row's slice closes here
              left[mt][h] = slice[mt][h];
              #pragma unroll
              for (int nt = 0; nt < NT; ++nt)
                #pragma unroll
                for (int e = 0; e < 2; ++e) {
                  float& x = acc[mt][nt][2 * h + e];
                  sum[mt][nt][2 * h + e] += __half2float(__float2half_rn(x));
                  x = 0.f;
                }
            }
          }
      }
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = m0 + wm * WM + mt * 16 + g + 8 * h;
          const float z = pad[mt][h] ? 0.f : -0.f;              // empty slices add +0 (-0 adds as nothing)
          if (row >= M)
            continue;
          #pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            const int col = n0 + wn * WN + nt * 8 + 2 * t;
            if (col < N)
              *reinterpret_cast<__half2*>(C + (size_t)row * N + col) =
                __floats2half2_rn(sum[mt][nt][2 * h] + z, sum[mt][nt][2 * h + 1] + z);
          }
        }
#endif
    }

    template <int TM, int TN, int WARPS_M = 2>
    inline void tsg_launch(const __half* a, const __half* w, __half* c, int m, int n, int k,
                           const SplitGroups& groups, cudaStream_t stream) {
      constexpr int smem = tsg_stages * (TM + TN) * tsg_pitch * sizeof (__half);
      static const bool configured = cudaFuncSetAttribute(tiled_split_gemm_kernel<TM, TN, WARPS_M>,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          smem) == cudaSuccess;
      (void)configured;
      const dim3 grid((n + TN - 1) / TN, (m + TM - 1) / TM);
      tiled_split_gemm_kernel<TM, TN, WARPS_M><<<grid, 128, smem, stream>>>(a, w, c, m, n, k, groups);
    }

    // One group of m rows, one chain over all of k (cuBLAS's arithmetic for the row-independent products).
    inline SplitGroups tsg_chain(int m, int k) {
      SplitGroups groups{};
      groups.count = 1;
      groups.row_end[0] = m;
      groups.slice[0] = k;
      groups.slices[0] = 1;
      return groups;
    }

  }
}

#pragma once

// One product C = A W^T (fp16, f32 accumulation) for the rows of several batches decoded together
// (cuda/clip_groups.h) when cuBLAS's arithmetic for a batch depends on its row count, as the Whisper decoder's
// second feed-forward (1280 x 5120) on the L40S: a batch of m rows runs split-K in S slices of L (each slice one
// mma.sync m16n8k16 chain over its k in increasing 16-groups, rounded to half; the slices summed in order in
// fp32; out = half(sum)), S and L chosen by m (grouped_split_gemm.cc). Here every row gets its own batch's S and
// L in one pass over the weights: a warp streams its 16 columns once for all rows (hmma_gemm_kernel.cuh's
// pipeline) and closes a row's slice where that row's batch would. Slice ends must be multiples of 64. As in
// hmma_probe.cu's candidates (which match cuBLAS bit for bit): the first slice is the sum itself (the sum starts
// at -0, the one value that adds as nothing, so -0 stays -0) and slices past k add +0.

#include <cstdint>
#include <vector>

#include <cuda_fp16.h>

namespace ctranslate2 {
  namespace cuda {

    constexpr int gsg_max_groups = 16, gsg_stages = 3, gsg_kstep = 64, gsg_pitch = 72;

    struct SplitGroups {
      int count;
      int row_end[gsg_max_groups];      // groups' rows: [row_end[g - 1], row_end[g])
      int slice[gsg_max_groups];        // each group's slice length (k for one slice)
      int slices[gsg_max_groups];       // and number of slices (more than k needs: the rest add +0)
    };

    __device__ __forceinline__ void gsg_ldm4(unsigned* r, const __half* p) {
      const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    }

    __device__ __forceinline__ void gsg_cp16(__half* dst, const __half* src, bool valid) {
#if __CUDA_ARCH__ >= 800
      const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(valid ? 16 : 0));
#endif
    }

    __device__ __forceinline__ void gsg_mma(float* d, const unsigned* a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
    }

    __device__ __forceinline__ int gsg_group_of(const SplitGroups& g, int row) {
      for (int i = 0; i < g.count; ++i)
        if (row < g.row_end[i])
          return i;
      return -1;
    }

    // Block: 16 columns, all M rows (MT 16-row tiles) shared by Warps warps (warp w: tiles w, w + Warps, ...), the
    // block's weights and all rows of A staged once a 64-k step; A [M x K], W [N x K], C [M x N], row-major.
    template <int MT, int Warps>
    __global__ void __launch_bounds__(Warps * 32)
    grouped_split_gemm_kernel(const __half* A, const __half* W, __half* C, int M, int N, int K, SplitGroups groups) {
#if __CUDA_ARCH__ >= 800
      constexpr int TPW = (MT + Warps - 1) / Warps;         // row tiles a warp
      extern __shared__ __align__(16) unsigned char gsg_smem[];
      __half* st = reinterpret_cast<__half*>(gsg_smem);
      constexpr int rows = MT * 16 + 16, stage = rows * gsg_pitch;
      const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, lr = lane % 8, lm = lane / 8;
      const int n0 = blockIdx.x * 16, steps = K / gsg_kstep;
      const int g = lane / 4, t = lane % 4;
      int slice[TPW][2];                                    // the slice length of this thread's two rows a tile
      bool pad[TPW][2];                                     // their batch has slices past k
      #pragma unroll
      for (int i = 0; i < TPW; ++i)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int mt = warp + i * Warps;
          const int grp = mt < MT ? gsg_group_of(groups, mt * 16 + g + 8 * h) : -1;
          slice[i][h] = grp < 0 ? 0 : groups.slice[grp];
          pad[i][h] = grp >= 0 && groups.slices[grp] > (K + groups.slice[grp] - 1) / groups.slice[grp];
        }
      auto load = [&](int s) {
        __half* dst = st + (s % gsg_stages) * stage;
        for (int v = threadIdx.x; v < rows * 8; v += Warps * 32) {
          const int r = v / 8, c = (v % 8) * 8;
          const bool is_a = r < MT * 16;
          const int row = is_a ? r : n0 + r - MT * 16;
          const bool valid = row < (is_a ? M : N);
          gsg_cp16(dst + r * gsg_pitch + c, (is_a ? A : W) + (size_t)(valid ? row : 0) * K + s * gsg_kstep + c, valid);
        }
      };
      for (int s = 0; s < gsg_stages - 1; ++s) {
        if (s < steps)
          load(s);
        asm volatile("cp.async.commit_group;");
      }
      float acc[2][TPW][4] = {}, sum[2][TPW][4];
      #pragma unroll
      for (int j = 0; j < 2; ++j)
        #pragma unroll
        for (int i = 0; i < TPW; ++i)
          #pragma unroll
          for (int e = 0; e < 4; ++e)
            sum[j][i][e] = -0.f;
      for (int s = 0; s < steps; ++s) {
        asm volatile("cp.async.wait_group %0;" :: "n"(gsg_stages - 2));
        __syncthreads();
        if (s + gsg_stages - 1 < steps)
          load(s + gsg_stages - 1);
        asm volatile("cp.async.commit_group;");
        const __half* base = st + (s % gsg_stages) * stage;
        #pragma unroll
        for (int q = 0; q < gsg_kstep / 16; ++q) {
          unsigned b[4];
          gsg_ldm4(b, base + (MT * 16 + (lm / 2) * 8 + lr) * gsg_pitch + q * 16 + (lm % 2) * 8);
          #pragma unroll
          for (int i = 0; i < TPW; ++i) {
            const int mt = warp + i * Warps;
            if (mt < MT) {
              unsigned a[4];
              gsg_ldm4(a, base + (mt * 16 + (lm % 2) * 8 + lr) * gsg_pitch + q * 16 + (lm / 2) * 8);
              gsg_mma(acc[0][i], a, b[0], b[1]);
              gsg_mma(acc[1][i], a, b[2], b[3]);
            }
          }
        }
        const int k_end = (s + 1) * gsg_kstep;
        #pragma unroll
        for (int i = 0; i < TPW; ++i)
          #pragma unroll
          for (int h = 0; h < 2; ++h)
            if (slice[i][h] > 0 && (k_end % slice[i][h] == 0 || k_end == K))   // this row's slice closes here
              #pragma unroll
              for (int j = 0; j < 2; ++j)
                #pragma unroll
                for (int e = 0; e < 2; ++e) {
                  float& a = acc[j][i][2 * h + e];
                  sum[j][i][2 * h + e] += __half2float(__float2half_rn(a));
                  a = 0.f;
                }
      }
      #pragma unroll
      for (int j = 0; j < 2; ++j)
        #pragma unroll
        for (int i = 0; i < TPW; ++i)
          #pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = (warp + i * Warps) * 16 + g + 8 * h, col = n0 + j * 8 + 2 * t;
            const float z = pad[i][h] ? 0.f : -0.f;           // empty slices add +0 (-0 adds as nothing)
            if (row < M && col < N)
              *reinterpret_cast<__half2*>(C + (size_t)row * N + col) =
                __floats2half2_rn(sum[j][i][2 * h] + z, sum[j][i][2 * h + 1] + z);
          }
#endif
    }

    // cuBLAS 12.9.2's second feed-forward of the Whisper decoder (1280 x 5120) on the L40S, by rows: split-K in
    // `slices` slices of `slice` k (hmma_probe.cu rows 17..48, ffn2_probe.cu rows 2..16, 4 fills each, no
    // mismatch; 5.10.2026). One row runs another kernel (gemv): no split known.
    inline bool gsg_split_of(int64_t rows, int& slice, int& slices) {
      if (rows >= 2 && rows <= 16) { slice = 320; slices = 16; }
      else if (rows >= 17 && rows <= 27) { slice = 768; slices = 8; }
      else if ((rows >= 28 && rows <= 34) || (rows >= 45 && rows <= 48)) { slice = 1728; slices = 3; }
      else if (rows >= 35 && rows <= 44) { slice = 5120; slices = 1; }
      else return false;
      return true;
    }

    template <int MT>
    void gsg_launch(const __half* a, const __half* w, __half* c, int m, int n, int k, const SplitGroups& groups,
                    cudaStream_t stream) {
      constexpr int warps = MT < 4 ? MT : 4;
      constexpr int smem = gsg_stages * (MT * 16 + 16) * gsg_pitch * sizeof (__half);
      static const bool configured = cudaFuncSetAttribute(grouped_split_gemm_kernel<MT, warps>,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          smem) == cudaSuccess;
      (void)configured;
      grouped_split_gemm_kernel<MT, warps><<<(n + 15) / 16, warps * 32, smem, stream>>>(a, w, c, m, n, k, groups);
    }

    // The second feed-forward for groups of rows (no device check): false, nothing launched, when a group's rows
    // have no known split, there are more than 160 rows or more than gsg_max_groups groups, or k is no multiple
    // of 64.
    inline bool gsg_run(const __half* a, const __half* w, __half* c, int n, int k,
                        const std::vector<int64_t>& group_rows, cudaStream_t stream) {
      if (group_rows.empty() || group_rows.size() > static_cast<size_t>(gsg_max_groups) || k % gsg_kstep != 0)
        return false;
      SplitGroups groups{};
      int m = 0;
      for (const int64_t rows : group_rows) {
        int slice = 0, slices = 0;
        if (!gsg_split_of(rows, slice, slices))
          return false;
        m += static_cast<int>(rows);
        groups.row_end[groups.count] = m;
        groups.slice[groups.count] = slice;
        groups.slices[groups.count] = slices;
        ++groups.count;
      }
      switch ((m + 15) / 16) {
      case 1: gsg_launch<1>(a, w, c, m, n, k, groups, stream); break;
      case 2: gsg_launch<2>(a, w, c, m, n, k, groups, stream); break;
      case 3: gsg_launch<3>(a, w, c, m, n, k, groups, stream); break;
      case 4: gsg_launch<4>(a, w, c, m, n, k, groups, stream); break;
      case 5: gsg_launch<5>(a, w, c, m, n, k, groups, stream); break;
      case 6: gsg_launch<6>(a, w, c, m, n, k, groups, stream); break;
      case 7: gsg_launch<7>(a, w, c, m, n, k, groups, stream); break;
      case 8: gsg_launch<8>(a, w, c, m, n, k, groups, stream); break;
      case 9: gsg_launch<9>(a, w, c, m, n, k, groups, stream); break;
      case 10: gsg_launch<10>(a, w, c, m, n, k, groups, stream); break;
      default: return false;
      }
      return true;
    }

  }
}

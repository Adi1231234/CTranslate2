#pragma once

// What tiled_split_gemm.cuh's kernel and its callers share: the groups of rows with their split of k, the
// instructions it is made of (the standard mma.sync m16n8k16 layout, as hmma_gemm_kernel.cuh), and cuBLAS
// 12.9.2's split of the Whisper decoder's second feed-forward on the L40S by rows.
// A batch of m rows runs split-K in S slices of L (each slice one mma.sync m16n8k16 chain over its k in increasing
// 16-groups, rounded to half; the slices summed in order in fp32; out = half(sum)), S and L chosen by m. As in
// hmma_probe.cu's candidates (which match cuBLAS bit for bit): the first slice is the sum itself (the sum starts at
// -0, the one value that adds as nothing, so -0 stays -0) and slices past k add +0. Slice ends: multiples of 32.

#include <cstdint>

#include <cuda_fp16.h>

namespace ctranslate2 {
  namespace cuda {

    constexpr int gsg_max_groups = 16, gsg_max_rows = 320;

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

  }
}

#pragma once

// Shapes, row places and the mma of exact_attention.cuh.

#include "softmax_kernels.cuh"
#include "exact_attention_layout.cuh"

namespace at {
  namespace native {

    constexpr int ea_warps = 8, ea_rows = 16, ea_depth = 64;
    constexpr int ea_lanes0 = 33, ea_part2 = 8 * ea_lanes0 * 4;   // slot stride and size of the first 1024 values

    __device__ __forceinline__ int ea_slot(int i, int tail_lanes) {   // key i -> its place in a stored row
      const int part = i >= 1024, j = i - 1024 * part, lanes = part ? tail_lanes : ea_lanes0;
      return ea_part2 * part + ((j % 32) / 4 * lanes + j / 32) * 4 + j % 4;
    }

    __device__ __forceinline__ void ea_mma(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, uint2 b) {
#if __CUDA_ARCH__ >= 800
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b.x), "r"(b.y));
#endif
    }

    // Whisper's encoder shape (m = n = 1500) as compile-time values, so the loops and places fold to constants.
    template <int N>
    struct ea_shape {
      static constexpr int tiles = (N + 7) / 8, groups = 2 + (N - N % 64) / 16, residue = N % 64;
      static constexpr int tail_lanes = (N - 1024 + 31) / 32;
      static constexpr int pitch = (ea_part2 + tail_lanes * 32 + 59) / 64 * 64 + 4;   // halves per row, 4 mod 64
      static constexpr int row_tiles = (N + ea_rows - 1) / ea_rows;                   // blocks' work per entry
    };

  }
}

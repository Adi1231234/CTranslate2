#pragma once

// slot_attention.cuh's shapes and device helpers (the parts' rows, the recipes' sums, the mma).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda/partial_sums.cuh"
#include "cuda/selfattn_recipes.h"
#include "cuda/slot_attention.h"

namespace ctranslate2 {
  namespace cuda {

    constexpr int sa_capacity = 448, sa_depth = 64, sa_rows = 5, sa_heads = 20;
    constexpr int sa_scores_count = sizeof (selfattn_scores_recipes) / sizeof (selfattn_scores_recipes[0]);
    constexpr int sa_output_count = sizeof (selfattn_output_recipes) / sizeof (selfattn_output_recipes[0]);

    static __device__ __forceinline__ float hf(__half x) {
      return __half2float(x);
    }

    // sum over the 64 dims of k[d] q[d] with a recipe's partials (selfattn_recipes.h).
    template <int T, int W, bool CONTIGUOUS, int TREE>
    static __device__ __forceinline__ float dot64(const __half* k, const float* q) {
      float s[T];
      #pragma unroll
      for (int r = 0; r < T; ++r)
        s[r] = 0.f;
      if (CONTIGUOUS) {
        constexpr int chunk = (sa_depth + T - 1) / T;
        #pragma unroll
        for (int r = 0; r < T; ++r)
          #pragma unroll
          for (int d = r * chunk; d < (r + 1) * chunk && d < sa_depth; ++d)
            s[r] = fmaf(hf(k[d]), q[d], s[r]);
      } else {
        #pragma unroll
        for (int base = 0; base < sa_depth; base += T * W)
          #pragma unroll
          for (int r = 0; r < T; ++r)
            #pragma unroll
            for (int u = 0; u < W; ++u)
              if (base + r * W + u < sa_depth)
                s[r] = fmaf(hf(k[base + r * W + u]), q[base + r * W + u], s[r]);
      }
      return combine<T>(s, TREE);
    }

    template <int CODE>
    static __device__ __forceinline__ float scores_dot(int code, const __half* k, const float* q) {
      if constexpr (CODE >= sa_scores_count) {
        return 0.f;
      } else {
        if (code == CODE) {
          constexpr SelfAttnRecipe r = selfattn_scores_recipes[CODE];
          if constexpr (r.kind == 0)
            return dot64<r.partials, r.vector, r.contiguous != 0, r.tree>(k, q);
          else
            return 0.f;
        }
        return scores_dot<CODE + 1>(code, k, q);
      }
    }

    // Row b's head h (its positions), or for a position under `shared` the shared row's.
    static __device__ __forceinline__ const __half* row_base(const void* cache, int row, int head, int heads,
                                                             int capacity) {
      return static_cast<const __half*>(cache) + (static_cast<size_t>(row) * heads + head) * capacity * sa_depth;
    }

    static __device__ __forceinline__ const __half* keys_at(const SlotAttention& part, int b, int i, int h, int heads) {
      return i < part.shared ? row_base(part.shared_keys, 0, h, heads, part.capacity)
                             : row_base(part.keys, b, h, heads, part.capacity);
    }

    static __device__ __forceinline__ const __half* values_at(const SlotAttention& part, int b, int i, int h,
                                                              int heads) {
      return i < part.shared ? row_base(part.shared_values, 0, h, heads, part.capacity)
                             : row_base(part.values, b, h, heads, part.capacity);
    }

    static __device__ __forceinline__ unsigned pair(const __half* p) {
      return *reinterpret_cast<const unsigned*>(p);
    }

    static __device__ __forceinline__ unsigned pack(__half lo, __half hi) {
      return unsigned(__half_as_ushort(lo)) | (unsigned(__half_as_ushort(hi)) << 16);
    }

    static __device__ __forceinline__ void mma16816(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                                    unsigned b0, unsigned b1) {
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                   "{%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

  }
}

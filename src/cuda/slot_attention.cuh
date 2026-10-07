#pragma once

// slot_attention.h's kernels and their launches (a header of its own so that tools/turing/kernels/
// slot_attention_check.cu runs these very kernels against cuBLAS).

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

    // Scores of the parts with a partials recipe: a thread per (position, part, head), every beam.
    static __global__ void slot_scores_partials(const SlotAttention* parts, const __half* queries, int heads, float scale) {
      const SlotAttention part = parts[blockIdx.z];
      const int code = part.scores_recipe;
      if (part.scores_mma)
        return;                                              // slot_scores_mma's
      __shared__ float qs[sa_rows][sa_depth];
      const int h = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
      for (int x = threadIdx.x; x < part.rows * sa_depth; x += blockDim.x)
        qs[x / sa_depth][x % sa_depth] =
          hf(queries[(static_cast<size_t>(part.row_begin + x / sa_depth) * heads + h) * sa_depth + x % sa_depth]);
      __syncthreads();
      if (i >= part.time)
        return;
      __half* scores = static_cast<__half*>(part.scores);
      __half kv[sa_depth];
      const auto load = [&](int b) {
        const __half* k = keys_at(part, b, i, h, heads) + static_cast<size_t>(i) * sa_depth;
        #pragma unroll
        for (int v = 0; v < sa_depth / 8; ++v)
          *reinterpret_cast<uint4*>(kv + 8 * v) = __ldg(reinterpret_cast<const uint4*>(k) + v);
      };
      const bool prompt = i < part.shared;                   // the same key for every beam: read once
      if (prompt)
        load(0);
      for (int b = 0; b < part.rows; ++b) {
        if (!prompt)
          load(b);
        const float sum = scores_dot<0>(code, kv, qs[b]);
        scores[(static_cast<size_t>(b) * heads + h) * part.time + i] = __float2half_rn(scale * sum);
      }
    }

    // Scores of the parts with the mma recipe: a warp per (16 positions, head, part), each beam an mma.sync m16n8k16
    // chain over the dims, the positions the rows of A and the beam's query column 0 of B.
    static __global__ void slot_scores_mma(const SlotAttention* parts, const __half* queries, int heads, float scale) {
      const SlotAttention part = parts[blockIdx.z];
      const int lane = threadIdx.x, g = lane >> 2, t = lane & 3, i0 = blockIdx.x * 16, h = blockIdx.y;
      if (!part.scores_mma || i0 >= part.time)
        return;
      __half* scores = static_cast<__half*>(part.scores);
      if (i0 + 16 <= part.shared && part.rows <= 8) {
        // 16 prompt positions, alike in every beam: beam c's query column c of B, one chain for them all (each of
        // the mma's outputs has its own row of A and column of B: beam c's are those of its own chain below).
        const __half* kb = row_base(part.shared_keys, 0, h, heads, part.capacity);
        const auto key_pair = [&](int i, int d) { return pair(kb + static_cast<size_t>(i) * sa_depth + d); };
        const __half* q = g < part.rows ? queries + (static_cast<size_t>(part.row_begin + g) * heads + h) * sa_depth
                                        : nullptr;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int d0 = 0; d0 < sa_depth; d0 += 16) {
          const unsigned b0 = q ? pair(q + d0 + 2 * t) : 0u, b1 = q ? pair(q + d0 + 8 + 2 * t) : 0u;
          mma16816(acc, key_pair(i0 + g, d0 + 2 * t), key_pair(i0 + g + 8, d0 + 2 * t),
                   key_pair(i0 + g, d0 + 8 + 2 * t), key_pair(i0 + g + 8, d0 + 8 + 2 * t), b0, b1);
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {                        // acc: (position g (+8), beam 2t (+1))
          const int b = 2 * t + (e & 1);
          if (b < part.rows)
            scores[(static_cast<size_t>(b) * heads + h) * part.time + i0 + g + 8 * (e >> 1)] =
              __float2half_rn(scale * acc[e]);
        }
        return;
      }
      for (int b = 0; b < part.rows; ++b) {
        const auto key_pair = [&](int i, int d) {
          return i < part.time ? pair(keys_at(part, b, i, h, heads) + static_cast<size_t>(i) * sa_depth + d) : 0u;
        };
        const __half* q = queries + (static_cast<size_t>(part.row_begin + b) * heads + h) * sa_depth;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int d0 = 0; d0 < sa_depth; d0 += 16) {
          const unsigned b0 = g == 0 ? pair(q + d0 + 2 * t) : 0u, b1 = g == 0 ? pair(q + d0 + 8 + 2 * t) : 0u;
          mma16816(acc, key_pair(i0 + g, d0 + 2 * t), key_pair(i0 + g + 8, d0 + 2 * t),
                   key_pair(i0 + g, d0 + 8 + 2 * t), key_pair(i0 + g + 8, d0 + 8 + 2 * t), b0, b1);
        }
        if (t == 0) {                                        // column 0: acc[0] row g, acc[2] row g + 8
          __half* row = scores + (static_cast<size_t>(b) * heads + h) * part.time;
          if (i0 + g < part.time)
            row[i0 + g] = __float2half_rn(scale * acc[0]);
          if (i0 + g + 8 < part.time)
            row[i0 + g + 8] = __float2half_rn(scale * acc[2]);
        }
      }
    }

    // Output of the parts with a partials recipe: a block per (32 dims, head, part), each beam on sa_out_lanes lanes
    // of its own summing the recipe's partials, so the beams read a prompt position's values at once (from L1 after
    // the first). A block a beam (8 lanes) read the prompt once a beam; a part's beams all on one lane set
    // (71642e85) made blocks five times longer (long32 w1: 106.7x against 136x).
    constexpr int sa_out_lanes = 4;
    template <int CODE, typename A, typename B>
    static __device__ __forceinline__ void output_partials(int code, float* sm, int x, int lane, int t, const A& a,
                                                           const B& b) {
      if constexpr (CODE < sa_output_count) {
        if (code == CODE) {
          constexpr SelfAttnRecipe r = selfattn_output_recipes[CODE];
          if constexpr (r.kind == 0)
            for (int q = lane; q < r.partials; q += sa_out_lanes)
              sm[q * 32 + x] = partial_sum<r.partials, r.vector, r.contiguous != 0>(q, t, a, b);
          return;
        }
        output_partials<CODE + 1>(code, sm, x, lane, t, a, b);
      }
    }

    template <int CODE>
    static __device__ __forceinline__ float output_combine(int code, const float* sm, int x) {
      if constexpr (CODE >= sa_output_count) {
        return 0.f;
      } else {
        if (code == CODE) {
          constexpr SelfAttnRecipe r = selfattn_output_recipes[CODE];
          if constexpr (r.kind == 0) {
            float s[r.partials];
            #pragma unroll
            for (int q = 0; q < r.partials; ++q)
              s[q] = sm[q * 32 + x];
            return combine<r.partials>(s, r.tree);
          } else {
            return 0.f;
          }
        }
        return output_combine<CODE + 1>(code, sm, x);
      }
    }

    static __global__ void slot_output_partials(const SlotAttention* parts, __half* out, int heads) {
      const SlotAttention part = parts[blockIdx.z];
      if (part.output_mma)
        return;                                              // the whole block
      __shared__ float sm[sa_rows][32 * 32];                // a beam's T x 32 partials, T at most 32
      const int x = threadIdx.x, lane = threadIdx.y % sa_out_lanes, b = threadIdx.y / sa_out_lanes;
      const int d = blockIdx.x * 32 + x, h = blockIdx.y;
      if (b < part.rows) {
        const __half* p = static_cast<const __half*>(part.scores) + (static_cast<size_t>(b) * heads + h) * part.time;
        const __half* v0 = row_base(part.shared_values, 0, h, heads, part.capacity) + d;
        const __half* vb = row_base(part.values, b, h, heads, part.capacity) + d;
        const int shared = part.shared;
        const auto pa = [&](int i) { return hf(p[i]); };
        const auto vv = [&](int i) { return hf((i < shared ? v0 : vb)[static_cast<size_t>(i) * sa_depth]); };
        output_partials<0>(part.output_recipe, sm[b], x, lane, part.time, pa, vv);
      }
      __syncthreads();
      if (lane == 0 && b < part.rows)
        out[(static_cast<size_t>(part.row_begin + b) * heads + h) * sa_depth + d] =
          __float2half_rn(output_combine<0>(part.output_recipe, sm[b], x));
    }

    // Output of the parts with the mma recipe: a warp per (8 dims, head, part), each beam an mma chain over the
    // positions in 16-groups from position 0 (zeros past t), the probabilities row 0 of A.
    static __global__ void slot_output_mma(const SlotAttention* parts, __half* out, int heads) {
      const SlotAttention part = parts[blockIdx.z];
      if (!part.output_mma)
        return;
      const int lane = threadIdx.x, g = lane >> 2, t = lane & 3, d0 = blockIdx.x * 8, h = blockIdx.y;
      const __half zero = __float2half(0.f);
      const auto probs = [&](int b) {
        return static_cast<const __half*>(part.scores) + (static_cast<size_t>(b) * heads + h) * part.time;
      };
      // The 16-groups of prompt positions first, alike in every beam: beam c's probabilities row c of A, one chain
      // for them all (each output has its own row of A and column of B); then each beam's chain goes on from its
      // row's sums, moved to row 0 (its own chain's).
      const int joint = part.rows <= 8 ? part.shared / 16 * 16 : 0;
      float rows[4] = {0.f, 0.f, 0.f, 0.f};
      if (joint > 0) {
        const __half* p = g < part.rows ? probs(g) : nullptr;
        const __half* v = row_base(part.shared_values, 0, h, heads, part.capacity);
        const auto vv = [&](int i, int d) { return v[static_cast<size_t>(i) * sa_depth + d]; };
        for (int i0 = 0; i0 < joint; i0 += 16) {
          const unsigned a0 = p ? pack(p[i0 + 2 * t], p[i0 + 2 * t + 1]) : 0u;
          const unsigned a2 = p ? pack(p[i0 + 8 + 2 * t], p[i0 + 9 + 2 * t]) : 0u;
          const unsigned b0 = pack(vv(i0 + 2 * t, d0 + g), vv(i0 + 2 * t + 1, d0 + g));
          const unsigned b1 = pack(vv(i0 + 8 + 2 * t, d0 + g), vv(i0 + 9 + 2 * t, d0 + g));
          mma16816(rows, a0, 0u, a2, 0u, b0, b1);
        }
      }
      for (int b = 0; b < part.rows; ++b) {
        const __half* p = probs(b);
        const auto pp = [&](int i) { return i < part.time ? p[i] : zero; };
        const auto vv = [&](int i, int d) {
          return i < part.time ? values_at(part, b, i, h, heads)[static_cast<size_t>(i) * sa_depth + d] : zero;
        };
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        if (joint > 0) {                                     // row b's sums (lanes 4b + t) to row 0 (lanes t)
          const float s0 = __shfl_sync(0xffffffffu, rows[0], 4 * b + t);
          const float s1 = __shfl_sync(0xffffffffu, rows[1], 4 * b + t);
          if (g == 0) {
            acc[0] = s0;
            acc[1] = s1;
          }
        }
        for (int i0 = joint; i0 < part.time; i0 += 16) {
          const unsigned a0 = g == 0 ? pack(pp(i0 + 2 * t), pp(i0 + 2 * t + 1)) : 0u;
          const unsigned a2 = g == 0 ? pack(pp(i0 + 8 + 2 * t), pp(i0 + 9 + 2 * t)) : 0u;
          const unsigned b0 = pack(vv(i0 + 2 * t, d0 + g), vv(i0 + 2 * t + 1, d0 + g));
          const unsigned b1 = pack(vv(i0 + 8 + 2 * t, d0 + g), vv(i0 + 9 + 2 * t, d0 + g));
          mma16816(acc, a0, 0u, a2, 0u, b0, b1);
        }
        if (g == 0)
          for (int c = 0; c < 2; ++c)
            out[(static_cast<size_t>(part.row_begin + b) * heads + h) * sa_depth + d0 + 2 * t + c] =
              __float2half_rn(acc[c]);
      }
    }

    inline void sa_scores_launch(const SlotAttention* parts, int count, const __half* q, int heads, int max_time,
                                 float scale, cudaStream_t stream) {
      if (count == 0)
        return;
      slot_scores_partials<<<dim3((max_time + 127) / 128, heads, count), 128, 0, stream>>>(parts, q, heads, scale);
      slot_scores_mma<<<dim3((max_time + 15) / 16, heads, count), 32, 0, stream>>>(parts, q, heads, scale);
    }

    inline void sa_output_launch(const SlotAttention* parts, int count, __half* out, int heads, cudaStream_t stream) {
      if (count == 0)
        return;
      slot_output_partials<<<dim3(sa_depth / 32, heads, count), dim3(32, sa_out_lanes * sa_rows), 0, stream>>>(
        parts, out, heads);
      slot_output_mma<<<dim3(sa_depth / 8, heads, count), 32, 0, stream>>>(parts, out, heads);
    }

  }
}

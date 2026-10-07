#pragma once

// slot_attention.cuh's output kernels (p v of every part) and their launch.

#include "cuda/slot_attention_common.cuh"

namespace ctranslate2 {
  namespace cuda {

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
          if constexpr (r.kind == 0) {
            constexpr int Q = r.partials > sa_out_lanes ? r.partials / sa_out_lanes : 1;
            float s[Q];
            partial_sums<r.partials, r.vector, r.contiguous != 0, Q>(lane, sa_out_lanes, t, a, b, s);
            #pragma unroll
            for (int k = 0; k < Q; ++k)
              if (lane + k * sa_out_lanes < r.partials)
                sm[(lane + k * sa_out_lanes) * 32 + x] = s[k];
          }
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

    inline void sa_output_launch(const SlotAttention* parts, int count, __half* out, int heads, cudaStream_t stream) {
      if (count == 0)
        return;
      slot_output_partials<<<dim3(sa_depth / 32, heads, count), dim3(32, sa_out_lanes * sa_rows), 0, stream>>>(
        parts, out, heads);
      slot_output_mma<<<dim3(sa_depth / 8, heads, count), 32, 0, stream>>>(parts, out, heads);
    }

  }
}

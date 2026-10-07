#pragma once

// slot_attention.cuh's scores kernels (scale q k^T of every part) and their launch.

#include "cuda/slot_attention_common.cuh"

namespace ctranslate2 {
  namespace cuda {

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

    inline void sa_scores_launch(const SlotAttention* parts, int count, const __half* q, int heads, int max_time,
                                 float scale, cudaStream_t stream) {
      if (count == 0)
        return;
      slot_scores_partials<<<dim3((max_time + 127) / 128, heads, count), 128, 0, stream>>>(parts, q, heads, scale);
      slot_scores_mma<<<dim3((max_time + 15) / 16, heads, count), 32, 0, stream>>>(parts, q, heads, scale);
    }

  }
}

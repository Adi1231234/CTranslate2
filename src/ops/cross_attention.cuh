#pragma once

// Whisper decoder cross-attention in one kernel: o = SoftMax(MatMul(q, k^T, alpha)) v for m queries (the beams of a
// clip, or the prompt at the first step) against the 1500 encoder positions of one (clip, head), with the
// arithmetic of the three ops it replaces on sm_120 with cuBLAS 12.9.2, which run as mma chains with the queries
// on the n side:
//   scores:  cutlass_80_wmma 32x32_64x1_tn with keys as the A operand: one m16n8k16 chain over the 64 dims from
//            zero (16 at a time in order), s = half(alpha * acc)
//   softmax: rows1024_row, the fork's replay of the legacy kernel's order (softmax_rows1024.cuh)
//   output:  cutlass_80_wmma 32x32_32x1_nn with the values (dims) as the A operand: one chain over the keys, 16 at
//            a time, the 32-key tiles' residue first (ca_group_start); o = half(acc)
// A block owns one (clip, head) and up to `rows` queries per pass (8 = the mma's n): its 4 warps compute the
// scores of 16-key tiles w, w + 4, ... into shared memory, run the softmax in place (a row per warp), then warp
// w chains the output's dims 16w..16w + 15 over all keys. Keys and values come in cross_attention_layout.cuh's
// fragment order; the output is written with the heads combined, [clip, query, head, dim].

#include "exact_attention_parts.cuh"
#include "cross_attention_layout.cuh"

namespace at {
  namespace native {

    constexpr int ca_warps = 4, ca_pitch = 1544;        // halves per score row: 772 words, 4 mod 32 banks

    // q: [entries][m][64] (entry = clip * heads + head), kf, vf: [entries][1504][64] fragments, o: [clips][m][heads][64].
    static __global__ void __launch_bounds__(ca_warps * 32)
    cross_attention_kernel(const __half* q, const uint4* kf, const uint4* vf, __half* o, int heads, int m,
                           int rows_per_pass, float alpha) {
      extern __shared__ __align__(16) unsigned char ca_smem[];
      __half* p = reinterpret_cast<__half*>(ca_smem);        // [rows_per_pass][ca_pitch] scores, probabilities
      const int entry = blockIdx.x, clip = entry / heads, head = entry % heads;
      const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
      const __half* qe = q + (size_t)entry * m * ca_depth;
      const uint4* ke = kf + (size_t)entry * ca_tiles * 4 * 32 + lane;
      const uint4* ve = vf + ((size_t)entry * 4 + warp) * ca_groups * 32 + lane;
      for (int j0 = 0; j0 < m; j0 += rows_per_pass) {
        const int rows = min(rows_per_pass, m - j0);
        unsigned b[4][2];                                    // query g's dims 16c + 2t.. (zero past the rows)
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          const __half* qr = qe + (size_t)(j0 + g) * ca_depth + 16 * c + 2 * t;
          b[c][0] = g < rows ? *reinterpret_cast<const unsigned*>(qr) : 0u;
          b[c][1] = g < rows ? *reinterpret_cast<const unsigned*>(qr + 8) : 0u;
        }
        #pragma unroll 4
        for (int T = warp; T < ca_tiles; T += ca_warps) {    // scores of keys 16T .. 16T + 15
          float d[4] = {0.f, 0.f, 0.f, 0.f};
          #pragma unroll
          for (int c = 0; c < 4; ++c) {
            const uint4 a = ke[(T * 4 + c) * 32];
            ea_mma(d, a.x, a.y, a.z, a.w, make_uint2(b[c][0], b[c][1]));
          }
          const int k0 = 16 * T + g, k1 = k0 + 8;             // d: (k0, 2t), (k0, 2t + 1), (k1, 2t), (k1, 2t + 1)
          #pragma unroll
          for (int h = 0; h < 2; ++h)
            if (2 * t + h < rows) {
              __half* row = p + (2 * t + h) * ca_pitch;
              if (k0 < ca_keys) row[k0] = __float2half_rn(alpha * d[h]);
              if (k1 < ca_keys) row[k1] = __float2half_rn(alpha * d[2 + h]);
            }
        }
        __syncthreads();
        for (int r = warp; r < rows; r += ca_warps) {        // softmax in place, a row per warp
          __half* row = p + r * ca_pitch;
          rows1024_row(row + 32 * lane, row + 1024 + 32 * lane, row + 32 * lane, row + 1024 + 32 * lane,
                       1, 1, ca_keys, lane);
        }
        __syncthreads();
        const __half* pr = p + g * ca_pitch;                 // B fragments: query g's probabilities
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll 4
        for (int G = 0; G < ca_groups; ++G) {
          const int s = ca_group_start(G) + 2 * t;
          const uint2 bb = g < rows ? make_uint2(*reinterpret_cast<const unsigned*>(pr + s),
                                                 *reinterpret_cast<const unsigned*>(pr + s + 8))
                                    : make_uint2(0u, 0u);
          const uint4 a = ve[G * 32];
          ea_mma(acc, a.x, a.y, a.z, a.w, bb);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {                        // acc: (dim g, query 2t), (g, 2t + 1), (g + 8, ..)
          const int query = 2 * t + i % 2, dim = 16 * warp + g + 8 * (i / 2);
          if (query < rows)
            o[(((size_t)clip * m + j0 + query) * heads + head) * ca_depth + dim] = __float2half_rn(acc[i]);
        }
        __syncthreads();                                     // before the next pass writes scores
      }
    }

  }
}

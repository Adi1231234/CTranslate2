#pragma once

// Whisper decoder cross-attention in one kernel: o = SoftMax(MatMul(q, k^T, alpha)) v for m queries (the beams of a
// clip, or the prompt) against the 1500 encoder positions of one (clip, head), read from the plain cache, with the
// arithmetic of the three ops it replaces on sm_120 with cuBLAS 12.9.2 (tools/turing/kernels/cross_sweep.cu):
//   scores:  cutlass_80_wmma ..._tn, keys as the A operand: one m16n8k16 chain over the 64 dims from zero,
//            s = half(alpha * acc)
//   softmax: rows1024_row, the fork's replay of the legacy kernel's order (softmax_rows1024.cuh)
//   output:  cutlass_80_wmma ..._nn, values (dims) as the A operand: one chain over the keys, 16 at a time, the
//            k tile's residue r first ([0, 16), .. up to r with zeros past it), then from r; o = half(acc).
//            r depends on the tile cuBLAS picks for the shape (cuda/cross_attention.h)
// A block owns one (clip, head) and up to `rows` queries per pass (8 = the mma's n): its 4 warps compute the scores
// of 16-key tiles w, w + 4, ... into shared memory, run the softmax in place (a row per warp), then warp w chains
// the output's dims 16w..16w + 15 over all keys. Its fragment row g holds dim 16w + 2g and row g + 8 dim
// 16w + 2g + 1 (an output's arithmetic does not depend on its row), so the value loads are 32-bit pairs of dims.

#include "cross_attention_q.cuh"

namespace at {
  namespace native {

    constexpr int ca_keys = 1500, ca_depth = 64, ca_tiles = (ca_keys + 15) / 16, ca_warps = 4;
    constexpr int ca_pitch = 1544;                           // halves per score row: 772 words, 4 mod 32 banks
    constexpr int ca_qpitch = 72;                            // halves per projected query row (phase 0)

    // The queries: q [entries][m][64], or (x non-null) projected by the block from x, w, bias with K inputs
    // (cross_attention_q.cuh).
    struct CaQueries {
      const __half* q;
      const __half* x;
      const __half* w;
      const __half* bias;
      int K;
    };

    // entry = clip * heads + head; k, v: [entries][1500][64]; o: [clips][m][heads][64].
    // 5 blocks per SM (registers for it): 8 clips x 20 heads = 160 blocks then fit the RTX 5060 Ti's 36 SMs at
    // once; at 4 (128 registers) a tail of 16 blocks ran after the rest.
    static __global__ void __launch_bounds__(ca_warps * 32, 5)
    cross_attention_kernel(CaQueries queries, const __half* k, const __half* v, __half* o, int heads, int m,
                           int rows_per_pass, int residue, float alpha, int ahead) {
      extern __shared__ __align__(16) unsigned char ca_smem[];
      __half* p = reinterpret_cast<__half*>(ca_smem);        // [rows_per_pass][ca_pitch] scores, probabilities
      __half* qs = p + rows_per_pass * ca_pitch;             // [rows_per_pass][ca_qpitch] projected queries
      const int entry = blockIdx.x, clip = entry / heads, head = entry % heads;
      const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
      const __half* ke = k + (size_t)entry * ca_keys * ca_depth;
      const __half* ve = v + (size_t)entry * ca_keys * ca_depth + 16 * warp + 2 * g;
      for (int j0 = 0; j0 < m; j0 += rows_per_pass) {
        const int rows = min(rows_per_pass, m - j0);
        if (queries.x) {
          // The projection waits on L2 more than it reads: the first half of the keys goes to L2 meanwhile.
          if (j0 == 0)
            for (int key = threadIdx.x; key < ca_keys / 2; key += ca_warps * 32)
              asm volatile("prefetch.global.L2 [%0];" :: "l"(ke + (size_t)key * ca_depth));
          ca_project_queries(queries.x, queries.w, queries.bias, qs, ca_qpitch, clip, head, m, j0, rows,
                             queries.K);
          __syncthreads();
        }
        const __half* qr0 = queries.x ? qs + g * ca_qpitch                               // query g's row
                                      : queries.q + ((size_t)entry * m + j0 + g) * ca_depth;
        unsigned b[4][2];                                    // query g's dims 16c + 2t.. (zero past the rows)
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          b[c][0] = g < rows ? ca_word(qr0 + 16 * c + 2 * t) : 0u;
          b[c][1] = g < rows ? ca_word(qr0 + 16 * c + 2 * t + 8) : 0u;
        }
        #pragma unroll 4
        for (int T = warp; T < ca_tiles; T += ca_warps) {    // scores of keys 16T .. 16T + 15
          const int ahead_key = 16 * (T + ca_warps * ahead) + lane;   // this warp's tile `ahead` steps on, into L2
          if (ahead && lane < 16 && ahead_key < ca_keys)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(ke + (size_t)ahead_key * ca_depth));
          const int k0 = 16 * T + g, k1 = k0 + 8;
          const __half* r0 = ke + (size_t)k0 * ca_depth + 2 * t;
          const __half* r1 = ke + (size_t)k1 * ca_depth + 2 * t;
          float d[4] = {0.f, 0.f, 0.f, 0.f};
          #pragma unroll
          for (int c = 0; c < 4; ++c)
            ea_mma(d, k0 < ca_keys ? ca_word(r0 + 16 * c) : 0u, k1 < ca_keys ? ca_word(r1 + 16 * c) : 0u,
                   k0 < ca_keys ? ca_word(r0 + 16 * c + 8) : 0u, k1 < ca_keys ? ca_word(r1 + 16 * c + 8) : 0u,
                   make_uint2(b[c][0], b[c][1]));
          #pragma unroll
          for (int h = 0; h < 2; ++h)                          // d: (k0, 2t), (k0, 2t + 1), (k1, 2t), (k1, 2t + 1)
            if (2 * t + h < rows) {
              __half* row = p + (2 * t + h) * ca_pitch;
              if (k0 < ca_keys) row[k0] = __float2half_rn(alpha * d[h]);
              if (k1 < ca_keys) row[k1] = __float2half_rn(alpha * d[2 + h]);
            }
        }
        // The memory is idle while the block finishes its scores and runs the softmax: the first values it will
        // read (keys 0 .. 4 * 128 - 1, a 128-byte row each) are fetched into L2 meanwhile.
        if (j0 == 0)
          #pragma unroll
          for (int i = 0; i < 4; ++i)
            asm volatile("prefetch.global.L2 [%0];"
                         :: "l"(v + ((size_t)entry * ca_keys + threadIdx.x + i * ca_warps * 32) * ca_depth));
        __syncthreads();
        for (int r = warp; r < rows; r += ca_warps) {        // softmax in place, a row per warp
          __half* row = p + r * ca_pitch;
          rows1024_row(row + 32 * lane, row + 1024 + 32 * lane, row + 32 * lane, row + 1024 + 32 * lane,
                       1, 1, ca_keys, lane);
        }
        __syncthreads();
        const __half* pr = p + g * ca_pitch;                 // B fragments: query g's probabilities
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        auto group = [&](int s, int end) {                  // keys s .. s + 15, zero from `end`
          const int ahead_key = s + 16 * ahead + 4 * warp + lane;   // group `ahead` on: 4 rows per warp into L2
          if (ahead && lane < 4 && ahead_key < ca_keys)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(v + ((size_t)entry * ca_keys + ahead_key) * ca_depth));
          const int i = s + 2 * t;
          unsigned x[4];                                    // V[i], V[i + 1], V[i + 8], V[i + 9] at dims 16w + 2g, +1
          #pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int key = i + (e & 1) + 8 * (e >> 1);
            x[e] = key < end ? ca_word(ve + (size_t)key * ca_depth) : 0u;
          }
          const uint2 bb = g < rows ? make_uint2(i < end ? ca_word(pr + i) : 0u, i + 8 < end ? ca_word(pr + i + 8) : 0u)
                                    : make_uint2(0u, 0u);
          ea_mma(acc, __byte_perm(x[0], x[1], 0x5410), __byte_perm(x[0], x[1], 0x7632),
                 __byte_perm(x[2], x[3], 0x5410), __byte_perm(x[2], x[3], 0x7632), bb);
        };
        for (int s = 0; s < residue; s += 16)
          group(s, residue);
        #pragma unroll 4
        for (int s = residue; s < ca_keys; s += 16)
          group(s, ca_keys);
        #pragma unroll
        for (int h = 0; h < 2; ++h) {                        // acc: (dim 2g, query 2t), (2g, 2t + 1), (2g + 1, ..)
          const int query = 2 * t + h;
          if (query < rows)
            *reinterpret_cast<__half2*>(o + (((size_t)clip * m + j0 + query) * heads + head) * ca_depth + 16 * warp
                                        + 2 * g) = __floats2half2_rn(acc[h], acc[2 + h]);
        }
        __syncthreads();                                     // before the next pass writes scores
      }
    }

  }
}

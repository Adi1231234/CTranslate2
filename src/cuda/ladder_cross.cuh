#pragma once

// The kernels of ladder_cross.h (a long recording's sampled ladder rows against their clip's memory), in a header
// so that tools/turing/kernels/ladder_cross_check.cu runs these very kernels against cuBLAS.

#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>

#include "cuda/partial_sums.cuh"

namespace ctranslate2 {
  namespace cuda {

    constexpr int lc_keys = 1500, lc_depth = 64, lc_max_rows = 32;

    struct LadderRows {
      int count;                             // rows listed
      int8_t row[lc_max_rows];               // the rows (indices into the queries and outputs)
      int8_t group[lc_max_rows];             // each one's group's rows: its cuBLAS call has group x heads entries
    };

    static __device__ __forceinline__ float hf(__half x) {
      return __half2float(x);
    }

    static __device__ __forceinline__ unsigned pair(const __half* p) {   // p[0], p[1] (4-byte aligned)
      return *reinterpret_cast<const unsigned*>(p);
    }

    // Scores of rows whose groups are 1 or 2 rows (cuBLAS's gemv): a thread per (key, row), blockIdx.z the row.
    __global__ void lc_scores_gemv(const __half* q, const __half* k, __half* scores, LadderRows rows, int heads,
                                   float alpha) {
      const int i = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y, r = rows.row[blockIdx.z];
      if (i >= lc_keys)
        return;
      const uint4* kv = reinterpret_cast<const uint4*>(k + (static_cast<size_t>(h) * lc_keys + i) * lc_depth);
      const uint4* qv = reinterpret_cast<const uint4*>(q + (static_cast<size_t>(r) * heads + h) * lc_depth);
      float part[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int v = 0; v < lc_depth / 8; ++v) {               // dims 8v .. 8v + 7 in order
        const uint4 a = __ldg(kv + v), b = __ldg(qv + v);
        const __half* ah = reinterpret_cast<const __half*>(&a);
        const __half* bh = reinterpret_cast<const __half*>(&b);
        #pragma unroll
        for (int u = 0; u < 8; ++u)
          part[u & 3] = fmaf(hf(ah[u]), hf(bh[u]), part[u & 3]);
      }
      part[0] += part[2];
      part[1] += part[3];
      part[0] += part[1];
      scores[(static_cast<size_t>(r) * heads + h) * lc_keys + i] = __float2half_rn(alpha * part[0]);
    }

    // Scores of rows whose groups are 3..5 rows (cuBLAS's tensor-core kernel): a warp per (16 keys, head), the keys
    // the rows of A and 8 rows' queries the columns of B, an mma.sync m16n8k16 chain over the dims.
    __global__ void lc_scores_mma(const __half* q, const __half* k, __half* scores, LadderRows rows, int heads,
                                  float alpha) {
      const int lane = threadIdx.x, g = lane >> 2, t = lane & 3, i0 = blockIdx.x * 16, h = blockIdx.y;
      const __half* kb = k + static_cast<size_t>(h) * lc_keys * lc_depth;
      const auto key_pair = [&](int i, int d) { return i < lc_keys ? pair(kb + static_cast<size_t>(i) * lc_depth + d)
                                                                   : 0u; };
      unsigned a[4][4];
      #pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int d0 = 16 * s;
        a[s][0] = key_pair(i0 + g, d0 + 2 * t);
        a[s][1] = key_pair(i0 + g + 8, d0 + 2 * t);
        a[s][2] = key_pair(i0 + g, d0 + 8 + 2 * t);
        a[s][3] = key_pair(i0 + g + 8, d0 + 8 + 2 * t);
      }
      for (int c0 = 0; c0 < rows.count; c0 += 8) {
        const int col = c0 + g;
        const __half* qc = col < rows.count
          ? q + (static_cast<size_t>(rows.row[col]) * heads + h) * lc_depth : nullptr;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
          const int d0 = 16 * s;
          const unsigned b0 = qc ? pair(qc + d0 + 2 * t) : 0u, b1 = qc ? pair(qc + d0 + 8 + 2 * t) : 0u;
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                       "{%0,%1,%2,%3};"
                       : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                       : "r"(a[s][0]), "r"(a[s][1]), "r"(a[s][2]), "r"(a[s][3]), "r"(b0), "r"(b1));
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {                        // C[key g (+8)][column 2t (+1)]
          const int i = i0 + g + 8 * (e / 2), c = c0 + 2 * t + e % 2;
          if (i < lc_keys && c < rows.count)
            scores[(static_cast<size_t>(rows.row[c]) * heads + h) * lc_keys + i] = __float2half_rn(alpha * acc[e]);
        }
      }
    }

    // Output arithmetic by a row's group size (cuBLAS's gemv for that batch): a group of one row sums 32 chunks of
    // 47 keys, a group of 5 key i in partial i % 4, groups of 2..4 key i in partial i % 16 (ladder_cross_probe).
    enum LcOutputKind { lc_one = 0, lc_five = 1, lc_other = 2 };
    constexpr int lc_kind_partials[3] = {32, 4, 16};
    // A block's rows at most (lc_output_blocks' cap; rows x T x 32 floats of shared memory). Blocks of up to 32 rows
    // (8b22a0f0) read a head's values once for them all, but a ladder's 25 rows then ran in 40 blocks on 142 SMs and
    // a ladder call took 12 s instead of 5.6 (long23 b0/b1 against a0); one row a block (6d5d2607) re-reads them a
    // row, 24% of the ladders' stream's GPU time (prof3). ladder_cross_check times caps 1..8.
    constexpr int lc_block_rows_max = 8;

    // Blocks of rows whose groups have the same arithmetic: each block reads its head's values once for its rows.
    struct LcOutputBlocks {
      int count;
      int8_t kind[lc_max_rows];
      int8_t first[lc_max_rows];                             // its rows: row[first .. first + rows)
      int8_t rows[lc_max_rows];
      int8_t row[lc_max_rows];
    };

    // The rows by arithmetic, in row order within each kind, in blocks of at most cap (<= lc_block_rows_max) of
    // them; smem_bytes: the launch's dynamic shared memory.
    inline LcOutputBlocks lc_output_blocks(const LadderRows& all, size_t& smem_bytes, int cap) {
      LcOutputBlocks blocks{};
      int placed = 0, most = 0;
      for (int kind = 0; kind < 3; ++kind) {
        for (int y = 0; y < all.count; ++y) {
          const int group = all.group[y];
          if ((group == 1 ? lc_one : group == 5 ? lc_five : lc_other) != kind)
            continue;
          if (blocks.count == 0 || blocks.kind[blocks.count - 1] != kind || blocks.rows[blocks.count - 1] == cap) {
            blocks.kind[blocks.count] = static_cast<int8_t>(kind);
            blocks.first[blocks.count] = static_cast<int8_t>(placed);
            blocks.rows[blocks.count] = 0;
            ++blocks.count;
          }
          blocks.row[placed++] = all.row[y];
          ++blocks.rows[blocks.count - 1];
        }
      }
      for (int b = 0; b < blocks.count; ++b)
        most = std::max(most, blocks.rows[b] * lc_kind_partials[blocks.kind[b]]);
      smem_bytes = static_cast<size_t>(most) * 32 * sizeof (float);
      return blocks;
    }

    template <int T, bool CONTIGUOUS, int R>
    static __device__ __forceinline__ void lc_output_rows(const __half* p, const __half* vd, __half* out,
                                                          const int8_t* row, int rows, int heads, int h, int d) {
      extern __shared__ float sm[];
      const auto pa = [&](int k, int i) { return hf(p[(static_cast<size_t>(row[k]) * heads + h) * lc_keys + i]); };
      const auto vb = [&](int, int i) { return hf(vd[static_cast<size_t>(i) * lc_depth]); };   // one clip: shared
      const auto store = [&](int k, float sum) {
        out[(static_cast<size_t>(row[k]) * heads + h) * lc_depth + d] = __float2half_rn(sum);
      };
      split_partials_rows<T, 1, CONTIGUOUS, 1, R>(sm, threadIdx.x, threadIdx.y, lc_keys, rows, lc_keys, pa, vb,
                                                  store);
    }

    // Output: a block per (32 dims, head, block of rows), its threads 32 dims x split_lanes lanes sharing each
    // output's partials (partial_sums.cuh); every row's sums are those of a block of its own (split_partials_rows).
    // Launch: grid (lc_depth / 32, heads, blocks.count), block (32, split_lanes), lc_output_blocks' shared memory.
    __global__ void lc_output(const __half* p, const __half* v, __half* out, LcOutputBlocks blocks, int heads) {
      const int d = blockIdx.x * 32 + threadIdx.x, h = blockIdx.y, b = blockIdx.z;
      const __half* vd = v + static_cast<size_t>(h) * lc_keys * lc_depth + d;
      const int8_t* row = blocks.row + blocks.first[b];
      const int rows = blocks.rows[b];
      constexpr int R = lc_block_rows_max;
      switch (blocks.kind[b]) {
      case lc_one:
        lc_output_rows<lc_kind_partials[lc_one], true, R>(p, vd, out, row, rows, heads, h, d);
        break;
      case lc_five:
        lc_output_rows<lc_kind_partials[lc_five], false, R>(p, vd, out, row, rows, heads, h, d);
        break;
      default:
        lc_output_rows<lc_kind_partials[lc_other], false, R>(p, vd, out, row, rows, heads, h, d);
      }
    }

  }
}

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
    constexpr bool lc_kind_contiguous[3] = {true, false, false};
    // A block per (32 dims, head, group): its rows (one cuBLAS call's, at most lc_kind_rows) each on lc_kind_lanes y
    // lanes, every lane summing partials y, y + lanes, ... of its row, so a group's rows read each value at the same
    // time (once from L2, the rest from L1) and no lane idles. A block per row with 8 lanes (6d5d2607) read a head's
    // values once a row (25 times a ladder) and left half its lanes idle at 4 partials (prof3: lc_output 24% of the
    // ladders' stream); a block of up to 32 rows one lane set (8b22a0f0) made a block 25 times longer.
    constexpr int lc_kind_rows[3] = {1, 5, 4};
    constexpr int lc_kind_lanes[3] = {8, 4, 8};

    // The groups of one kind: group g's rows are row[first[g] .. first[g] + rows[g]).
    struct LcGroups {
      int count;
      int8_t first[lc_max_rows];
      int8_t rows[lc_max_rows];
      int8_t row[lc_max_rows];
    };

    // The rows' groups by kind (consecutive rows with one group size each, as row_groups lists them).
    inline void lc_output_groups(const LadderRows& all, LcGroups (&kinds)[3]) {
      for (auto& k : kinds)
        k = LcGroups{};
      for (int y = 0; y < all.count;) {
        const int size = all.group[y];
        LcGroups& k = kinds[size == 1 ? lc_one : size == 5 ? lc_five : lc_other];
        const int n = static_cast<int>(k.count == 0 ? 0 : k.first[k.count - 1] + k.rows[k.count - 1]);
        k.first[k.count] = static_cast<int8_t>(n);
        k.rows[k.count] = static_cast<int8_t>(size);
        for (int r = 0; r < size; ++r)
          k.row[n + r] = all.row[y + r];
        ++k.count;
        y += size;
      }
    }

    // One partial of a row's output column: the products p(i) v(i) of its keys in increasing order from zero (a
    // chunk of ceil(1500 / T) keys, or keys r, r + T, ...), each with a fused multiply-add.
    template <int T, bool CONTIGUOUS>
    static __device__ __forceinline__ float lc_partial(const __half* pr, const __half* vd, int r) {
      float s = 0.f;
      if (CONTIGUOUS) {
        constexpr int chunk = (lc_keys + T - 1) / T;
        const int end = min(lc_keys, (r + 1) * chunk);
        for (int i = r * chunk; i < end; ++i)
          s = fmaf(hf(pr[i]), hf(vd[static_cast<size_t>(i) * lc_depth]), s);
      } else {
        for (int i = r; i < lc_keys; i += T)
          s = fmaf(hf(pr[i]), hf(vd[static_cast<size_t>(i) * lc_depth]), s);
      }
      return s;
    }

    // Output: grid (lc_depth / 32, heads, groups.count), block (32, lanes x rows) of a kind (lc_output_launch); the
    // T partials of each output combined by the tree from the halves (partial_sums.cuh: combine), as before.
    template <int KIND>
    __global__ void lc_output(const __half* p, const __half* v, __half* out, LcGroups groups, int heads) {
      constexpr int T = lc_kind_partials[KIND], L = lc_kind_lanes[KIND];
      __shared__ float sm[lc_kind_rows[KIND] * T * 32];
      const int x = threadIdx.x, lane = threadIdx.y % L, k = threadIdx.y / L;
      const int d = blockIdx.x * 32 + x, h = blockIdx.y, b = blockIdx.z;
      const int rows = groups.rows[b];
      const int row = k < rows ? groups.row[groups.first[b] + k] : 0;
      if (k < rows) {
        const __half* pr = p + (static_cast<size_t>(row) * heads + h) * lc_keys;
        const __half* vd = v + static_cast<size_t>(h) * lc_keys * lc_depth + d;
        for (int r = lane; r < T; r += L)
          sm[(k * T + r) * 32 + x] = lc_partial<T, lc_kind_contiguous[KIND]>(pr, vd, r);
      }
      __syncthreads();
      if (lane == 0 && k < rows) {
        float s[T];
        #pragma unroll
        for (int r = 0; r < T; ++r)
          s[r] = sm[(k * T + r) * 32 + x];
        out[(static_cast<size_t>(row) * heads + h) * lc_depth + d] = __float2half_rn(combine<T>(s, 1));
      }
    }

    // A launch for each kind the rows have.
    inline void lc_output_launch(const __half* p, const __half* v, __half* out, const LadderRows& all, int heads,
                                 cudaStream_t stream) {
      LcGroups kinds[3];
      lc_output_groups(all, kinds);
      const auto launch = [&](auto kernel, int kind) {
        if (kinds[kind].count > 0)
          kernel<<<dim3(lc_depth / 32, heads, kinds[kind].count), dim3(32, lc_kind_lanes[kind] * lc_kind_rows[kind]),
                   0, stream>>>(p, v, out, kinds[kind], heads);
      };
      launch(lc_output<lc_five>, lc_five);
      launch(lc_output<lc_other>, lc_other);
      launch(lc_output<lc_one>, lc_one);
    }

  }
}

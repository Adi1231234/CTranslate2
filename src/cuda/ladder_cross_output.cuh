#pragma once

// ladder_cross.cuh's output kernel (a ladder's rows' probabilities times their clip's memory values) and launch.

#include "cuda/ladder_cross_common.cuh"

namespace ctranslate2 {
  namespace cuda {

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

    // Output: grid (lc_depth / 32, heads, groups.count), block (32, lanes x rows) of a kind (lc_output_launch); the
    // T partials of each output combined by the tree from the halves (partial_sums.cuh: combine), as before.
    // Output: grid (1, heads, groups.count), block (32, lanes x rows) of a kind (lc_output_launch), lane x the dims
    // 2x and 2x + 1 (their values read as one pair); the T partials of each output combined by the tree from the
    // halves (partial_sums.cuh: combine), as before.
    template <int KIND>
    __global__ void lc_output(const __half* p, const __half* v, __half* out, LcGroups groups, int heads) {
      constexpr int T = lc_kind_partials[KIND], L = lc_kind_lanes[KIND];
      __shared__ float2 sm[lc_kind_rows[KIND] * T * 32];
      const int x = threadIdx.x, lane = threadIdx.y % L, k = threadIdx.y / L;
      const int h = blockIdx.y, b = blockIdx.z;
      const int rows = groups.rows[b];
      const int row = k < rows ? groups.row[groups.first[b] + k] : 0;
      if (k < rows) {
        const __half* pr = p + (static_cast<size_t>(row) * heads + h) * lc_keys;
        const __half2* vd = reinterpret_cast<const __half2*>(v + static_cast<size_t>(h) * lc_keys * lc_depth) + x;
        const auto pa = [&](int i) { return hf(pr[i]); };
        const auto vb = [&](int i) { return __half22float2(vd[static_cast<size_t>(i) * (lc_depth / 2)]); };
        constexpr int Q = T > L ? T / L : 1;
        float2 s[Q];
        partial_sums2<T, 1, lc_kind_contiguous[KIND], Q>(lane, L, lc_keys, pa, vb, s);
        #pragma unroll
        for (int q = 0; q < Q; ++q)
          if (lane + q * L < T)
            sm[(k * T + lane + q * L) * 32 + x] = s[q];
      }
      __syncthreads();
      if (lane == 0 && k < rows) {
        float s[T], t[T];
        #pragma unroll
        for (int r = 0; r < T; ++r) {
          s[r] = sm[(k * T + r) * 32 + x].x;
          t[r] = sm[(k * T + r) * 32 + x].y;
        }
        *reinterpret_cast<__half2*>(out + (static_cast<size_t>(row) * heads + h) * lc_depth + 2 * x) =
          __floats2half2_rn(combine<T>(s, 1), combine<T>(t, 1));
      }
    }

    // A launch for each kind the rows have.
    inline void lc_output_launch(const __half* p, const __half* v, __half* out, const LadderRows& all, int heads,
                                 cudaStream_t stream) {
      LcGroups kinds[3];
      lc_output_groups(all, kinds);
      const auto launch = [&](auto kernel, int kind) {
        if (kinds[kind].count > 0)
          kernel<<<dim3(1, heads, kinds[kind].count), dim3(lc_depth / 2, lc_kind_lanes[kind] * lc_kind_rows[kind]),
                   0, stream>>>(p, v, out, kinds[kind], heads);
      };
      launch(lc_output<lc_five>, lc_five);
      launch(lc_output<lc_other>, lc_other);
      launch(lc_output<lc_one>, lc_one);
    }

  }
}

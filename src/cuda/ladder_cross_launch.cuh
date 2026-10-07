#pragma once

// ladder_cross.cuh's launches: several ladders' rows planned into one launch per kernel (lc_plan), as ladder_cross.cu
// and tools/turing/kernels/ladder_cross_check.cu run them.

#include <vector>

#include "cuda/ladder_cross_output.cuh"
#include "cuda/ladder_cross_scores.cuh"

namespace ctranslate2 {
  namespace cuda {

    // One ladder: its clip's memory keys and values, its first row in the queries, scores and outputs, and its groups'
    // rows in order (each group one cuBLAS call of its own, so one arithmetic).
    struct LcLadder {
      const __half* k;
      const __half* v;
      int row_begin;
      std::vector<int> groups;
    };

    // A launch's lists: every ladder's rows by scores arithmetic (gemv for groups of 1..2 rows, the tensor-core chain
    // for 3..5) and its groups by output kind.
    struct LcPlan {
      LcParts parts;
      LadderRows gemv, mma;
      LcGroups kinds[3];
    };

    // The ladders in one launch, or false where they do not fit: more than lc_max_parts ladders, a ladder past
    // lc_part_rows rows, more than lc_max_rows rows in all, or a group of no recovered arithmetic (0 or past 5 rows).
    inline bool lc_plan(const std::vector<LcLadder>& ladders, LcPlan& plan) {
      plan = LcPlan{};
      if (ladders.empty() || ladders.size() > static_cast<size_t>(lc_max_parts))
        return false;
      int all = 0;
      for (size_t p = 0; p < ladders.size(); ++p) {
        const LcLadder& ladder = ladders[p];
        plan.parts.k[p] = ladder.k;
        plan.parts.v[p] = ladder.v;
        plan.parts.row_begin[p] = ladder.row_begin;
        plan.gemv.part_begin[p] = static_cast<int16_t>(plan.gemv.count);
        plan.mma.part_begin[p] = static_cast<int16_t>(plan.mma.count);
        int row = 0;
        for (const int size : ladder.groups) {
          if (size < 1 || size > 5 || row + size > lc_part_rows || all + size > lc_max_rows)
            return false;
          LadderRows& list = size <= 2 ? plan.gemv : plan.mma;
          LcGroups& kind = plan.kinds[size == 1 ? lc_one : size == 5 ? lc_five : lc_other];
          const int n = kind.count == 0 ? 0 : kind.first[kind.count - 1] + kind.rows[kind.count - 1];
          kind.first[kind.count] = static_cast<int8_t>(n);
          kind.rows[kind.count] = static_cast<int8_t>(size);
          kind.part[kind.count++] = static_cast<int8_t>(p);
          for (int r = 0; r < size; ++r) {
            list.row[list.count] = static_cast<int8_t>(row + r);
            list.group[list.count] = static_cast<int8_t>(size);
            list.part[list.count++] = static_cast<int8_t>(p);
            kind.row[n + r] = static_cast<int8_t>(row + r);
          }
          row += size;
          all += size;
        }
      }
      plan.parts.count = static_cast<int>(ladders.size());
      plan.gemv.part_begin[ladders.size()] = static_cast<int16_t>(plan.gemv.count);
      plan.mma.part_begin[ladders.size()] = static_cast<int16_t>(plan.mma.count);
      return true;
    }

    inline void lc_scores_launch(const LcPlan& plan, const __half* q, __half* scores, int heads, float alpha,
                                 cudaStream_t stream) {
      if (plan.gemv.count > 0)
        lc_scores_gemv<<<dim3((lc_keys + 127) / 128, heads, plan.gemv.count), 128, 0, stream>>>(
          q, scores, plan.parts, plan.gemv, heads, alpha);
      if (plan.mma.count > 0)
        lc_scores_mma<<<dim3((lc_keys + 15) / 16, heads, plan.parts.count), 32, 0, stream>>>(
          q, scores, plan.parts, plan.mma, heads, alpha);
    }

    // A launch for each kind the rows have.
    inline void lc_output_launch(const LcPlan& plan, const __half* p, __half* out, int heads, cudaStream_t stream) {
      const auto launch = [&](auto kernel, int kind) {
        if (plan.kinds[kind].count > 0)
          kernel<<<dim3(1, heads, plan.kinds[kind].count),
                   dim3(lc_depth / 2, lc_kind_lanes[kind] * lc_kind_rows[kind]), 0, stream>>>(
            p, out, plan.parts, plan.kinds[kind], heads);
      };
      launch(lc_output<lc_five>, lc_five);
      launch(lc_output<lc_other>, lc_other);
      launch(lc_output<lc_one>, lc_one);
    }

  }
}

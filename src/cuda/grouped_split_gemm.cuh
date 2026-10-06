#pragma once

// One product C = A W^T (fp16, f32 accumulation) for the rows of several batches decoded together
// (cuda/clip_groups.h) when cuBLAS's arithmetic for a batch depends on its row count, as the Whisper decoder's
// second feed-forward (1280 x 5120) on the L40S (split_gemm_common.cuh): every row gets its own batch's split in
// one pass, in tiled_split_gemm.cuh's kernel with 64 x 64 output tiles and 5 stages of 64 k in flight (round30:
// 60 us against 110 for 3 stages of 32 k at 5 groups of 40 rows, the batched path +1.3%). More groups or rows than
// one launch holds (a long recording's windows, one group each: 34-48 groups of 5 rows) go in several launches of
// consecutive groups; a row's arithmetic is its group's either way.

#include <vector>

#include "cuda/tiled_split_gemm.cuh"

namespace ctranslate2 {
  namespace cuda {

    // The second feed-forward for groups of rows (no device check): false, nothing launched, when a group's rows
    // have no known split or k is no multiple of 64. A group of one row (cuBLAS runs a gemv for it) gets one chain
    // over k as a placeholder: the caller recomputes it alone.
    inline bool gsg_run(const __half* a, const __half* w, __half* c, int n, int k,
                        const std::vector<int64_t>& group_rows, cudaStream_t stream) {
      constexpr int kstep = 64;                              // every split's slice is a multiple of it
      if (group_rows.empty() || k % kstep != 0)
        return false;
      std::vector<SplitGroups> launches(1);                  // each at most gsg_max_groups groups, gsg_max_rows rows
      std::vector<int> first_rows(1, 0);
      int row = 0;
      for (const int64_t rows : group_rows) {
        int slice = k, slices = 1;                           // one row: a placeholder
        if (rows != 1 && !gsg_split_of(rows, slice, slices))   // 2..48 rows
          return false;
        SplitGroups* groups = &launches.back();
        const int held = groups->count ? groups->row_end[groups->count - 1] : 0;
        if (groups->count == gsg_max_groups || held + rows > gsg_max_rows) {
          launches.emplace_back();
          first_rows.push_back(row);
          groups = &launches.back();
        }
        const int end = (groups->count ? groups->row_end[groups->count - 1] : 0) + static_cast<int>(rows);
        groups->row_end[groups->count] = end;
        groups->slice[groups->count] = slice;
        groups->slices[groups->count] = slices;
        ++groups->count;
        row += static_cast<int>(rows);
      }
      for (size_t i = 0; i < launches.size(); ++i) {
        const SplitGroups& groups = launches[i];
        const size_t first = static_cast<size_t>(first_rows[i]);
        tsg_launch<64, 64, 2, 5, kstep>(a + first * k, w, c + first * n, groups.row_end[groups.count - 1], n, k,
                                        groups, stream);
      }
      return true;
    }

  }
}

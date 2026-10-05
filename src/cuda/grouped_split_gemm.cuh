#pragma once

// One product C = A W^T (fp16, f32 accumulation) for the rows of several batches decoded together
// (cuda/clip_groups.h) when cuBLAS's arithmetic for a batch depends on its row count, as the Whisper decoder's
// second feed-forward (1280 x 5120) on the L40S (split_gemm_common.cuh): every row gets its own batch's split in
// one pass, in tiled_split_gemm.cuh's kernel with 64 x 64 output tiles and a 3-stage pipeline.

#include <vector>

#include "cuda/tiled_split_gemm.cuh"

namespace ctranslate2 {
  namespace cuda {

    // The second feed-forward for groups of rows (no device check): false, nothing launched, when a group's rows
    // have no known split, there are more than gsg_max_rows rows or more than gsg_max_groups groups, or k is no
    // multiple of 32.
    inline bool gsg_run(const __half* a, const __half* w, __half* c, int n, int k,
                        const std::vector<int64_t>& group_rows, cudaStream_t stream) {
      if (group_rows.empty() || group_rows.size() > static_cast<size_t>(gsg_max_groups) || k % tsg_kstep != 0)
        return false;
      SplitGroups groups{};
      int m = 0;
      for (const int64_t rows : group_rows) {
        int slice = 0, slices = 0;
        if (!gsg_split_of(rows, slice, slices))
          return false;
        m += static_cast<int>(rows);
        groups.row_end[groups.count] = m;
        groups.slice[groups.count] = slice;
        groups.slices[groups.count] = slices;
        ++groups.count;
      }
      if (m > gsg_max_rows)
        return false;
      tsg_launch<64, 64, 2, 3>(a, w, c, m, n, k, groups, stream);
      return true;
    }

  }
}

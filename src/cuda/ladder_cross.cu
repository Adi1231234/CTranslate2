#include "cuda/ladder_cross.h"

#include "cuda/clip_groups.h"
#include "cuda/ladder_cross.cuh"
#include "cuda/shared_memory_rows.h"
#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    static bool enabled() {
      static const bool on = read_bool_from_env("CT2_LADDER_CROSS") && cublas_verified_on(8, 9);
      return on;
    }

    // Every row's group size as shared_memory_rows.cu's products split the rows (for_each_clip_group, else one call),
    // or false when a group has no recovered arithmetic (more than 5 rows) or the rows are too many.
    static bool row_groups(dim_t rows, LadderRows& all) {
      if (rows < 1 || rows > lc_max_rows)
        return false;
      bool known = true;
      all.count = static_cast<int>(rows);
      const auto set = [&](dim_t first, dim_t count) {
        known = known && count >= 1 && count <= 5;
        for (dim_t r = first; r < first + count; ++r) {
          all.row[r] = static_cast<int8_t>(r);
          all.group[r] = static_cast<int8_t>(count);
        }
      };
      if (!for_each_clip_group(rows, set))
        set(0, rows);
      return known;
    }

    bool ladder_cross_scores(const SharedMemoryRows& rows, const __half* q, const __half* k, __half* scores,
                             dim_t heads, dim_t keys, dim_t depth, float alpha) {
      LadderRows all{};
      if (!enabled() || rows.clips != 1 || keys != lc_keys || depth != lc_depth || !row_groups(rows.rows, all))
        return false;
      LadderRows gemv{}, mma{};
      for (int y = 0; y < all.count; ++y) {
        LadderRows& list = all.group[y] <= 2 ? gemv : mma;
        list.row[list.count] = all.row[y];
        list.group[list.count] = all.group[y];
        ++list.count;
      }
      cudaStream_t stream = get_cuda_stream();
      const int h = static_cast<int>(heads);
      if (gemv.count > 0)
        lc_scores_gemv<<<dim3((lc_keys + 127) / 128, h, gemv.count), 128, 0, stream>>>(q, k, scores, gemv, h, alpha);
      if (mma.count > 0)
        lc_scores_mma<<<dim3((lc_keys + 15) / 16, h), 32, 0, stream>>>(q, k, scores, mma, h, alpha);
      return true;
    }

    bool ladder_cross_output(const SharedMemoryRows& rows, const __half* p, const __half* v, __half* out, dim_t heads,
                             dim_t keys, dim_t depth) {
      LadderRows all{};
      if (!enabled() || rows.clips != 1 || keys != lc_keys || depth != lc_depth || !row_groups(rows.rows, all))
        return false;
      lc_output_launch(p, v, out, all, static_cast<int>(heads), get_cuda_stream());
      return true;
    }

  }
}

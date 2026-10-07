#include "cuda/ladder_cross.h"

#include <stdexcept>

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

    // One clip's rows as shared_memory_rows.cu's products split them (for_each_clip_group, else one call), or false
    // when they have no recovered arithmetic.
    static bool one_ladder(const SharedMemoryRows& rows, const __half* k, const __half* v, dim_t keys, dim_t depth,
                           LcPlan& plan) {
      if (!enabled() || rows.clips != 1 || keys != lc_keys || depth != lc_depth || rows.rows < 1)
        return false;
      LcLadder ladder{k, v, 0, {}};
      if (!for_each_clip_group(rows.rows, [&](dim_t, dim_t count) { ladder.groups.push_back(static_cast<int>(count)); }))
        ladder.groups.push_back(static_cast<int>(rows.rows));
      return lc_plan({ladder}, plan);
    }

    bool ladder_cross_scores(const SharedMemoryRows& rows, const __half* q, const __half* k, __half* scores,
                             dim_t heads, dim_t keys, dim_t depth, float alpha) {
      LcPlan plan;
      if (!one_ladder(rows, k, nullptr, keys, depth, plan))
        return false;
      lc_scores_launch(plan, q, scores, static_cast<int>(heads), alpha, get_cuda_stream());
      return true;
    }

    bool ladder_cross_output(const SharedMemoryRows& rows, const __half* p, const __half* v, __half* out, dim_t heads,
                             dim_t keys, dim_t depth) {
      LcPlan plan;
      if (!one_ladder(rows, nullptr, v, keys, depth, plan))
        return false;
      lc_output_launch(plan, p, out, static_cast<int>(heads), get_cuda_stream());
      return true;
    }

    // The ladders in launches of at most lc_max_parts ladders and lc_max_rows rows, in order.
    template <typename F>
    static bool for_each_launch(const std::vector<LadderMemory>& ladders, F&& launch) {
      std::vector<LcLadder> batch;
      int rows = 0;
      LcPlan plan;
      const auto flush = [&]() {
        if (batch.empty())
          return true;
        if (!lc_plan(batch, plan))
          return false;
        launch(plan);
        batch.clear();
        rows = 0;
        return true;
      };
      for (const LadderMemory& m : ladders) {
        LcLadder ladder{m.keys, m.values, static_cast<int>(m.row_begin), {}};
        int count = 0;
        for (const dim_t g : m.groups) {
          ladder.groups.push_back(static_cast<int>(g));
          count += static_cast<int>(g);
        }
        if (batch.size() == static_cast<size_t>(lc_max_parts) || rows + count > lc_max_rows)
          if (!flush())
            return false;
        batch.push_back(std::move(ladder));
        rows += count;
      }
      return flush();
    }

    bool ladders_supported(const std::vector<LadderMemory>& ladders) {
      if (!enabled())
        return false;
      LcPlan plan;
      for (const LadderMemory& m : ladders) {                // each ladder fits a launch alone
        LcLadder ladder{m.keys, m.values, 0, {}};
        for (const dim_t g : m.groups)
          ladder.groups.push_back(static_cast<int>(g));
        if (!lc_plan({ladder}, plan))
          return false;
      }
      return true;
    }

    void ladders_cross_scores(const std::vector<LadderMemory>& ladders, const __half* q, __half* scores, dim_t heads,
                              float alpha) {
      cudaStream_t stream = get_cuda_stream();
      if (!for_each_launch(ladders, [&](const LcPlan& plan) {
            lc_scores_launch(plan, q, scores, static_cast<int>(heads), alpha, stream); }))
        throw std::logic_error("ladders_cross_scores: ladders without a recovered arithmetic");
    }

    void ladders_cross_output(const std::vector<LadderMemory>& ladders, const __half* p, __half* out, dim_t heads) {
      cudaStream_t stream = get_cuda_stream();
      if (!for_each_launch(ladders, [&](const LcPlan& plan) {
            lc_output_launch(plan, p, out, static_cast<int>(heads), stream); }))
        throw std::logic_error("ladders_cross_output: ladders without a recovered arithmetic");
    }

  }
}

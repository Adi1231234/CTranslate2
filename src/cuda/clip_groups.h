#pragma once

#include <vector>

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // The clips of one decoder batch that stand for several independent batches (WhisperOptions::group_size):
    // while a ClipGroupsScope is active on a thread, every cuBLAS product of that thread whose rows (or batch
    // entries) are a whole multiple of the clips, and the fused cross-attention, run once per group with that
    // group's rows only: the very call a batch of that group alone makes (same shape, same kernel, same bits).
    // The groups' calls follow each other, so after the first one the weights come from L2 instead of DRAM.
    // Clips are grouped contiguously in batch order; finished clips leave and the others keep their order.
    struct ClipGroups {
      std::vector<dim_t> clips;   // clips of each group still in the batch (0: the group has finished)
      dim_t total = 0;
    };

    // The calling thread's groups, or nullptr when no scope is active or fewer than two groups have clips.
    const ClipGroups* clip_groups();

    // Groups of `group_size` clips by original batch index (`batch_offset[i]` is the original index of the
    // batch's i-th clip, as the decoding loops keep it); no groups when group_size is 0.
    ClipGroups make_clip_groups(const std::vector<dim_t>& batch_offset, dim_t group_size);

    class ClipGroupsScope {
    public:
      explicit ClipGroupsScope(ClipGroups groups);
      ~ClipGroupsScope();
      ClipGroupsScope(const ClipGroupsScope&) = delete;
      ClipGroupsScope& operator=(const ClipGroupsScope&) = delete;
    private:
      ClipGroups _groups;
      const ClipGroups* _previous;
    };

    // Calls f(first unit, units) for each group with clips, where a unit is count / total rows or entries (the
    // rows of one clip); returns false, calling nothing, when no groups are active or count is no whole
    // multiple of the clips. The calls run with no groups active, so a product inside f is not split again.
    template <typename F>
    bool for_each_clip_group(dim_t count, F&& f);

    // Clears the calling thread's groups while it lives (for_each_clip_group's calls).
    class ClipGroupsPause {
    public:
      ClipGroupsPause();
      ~ClipGroupsPause();
      ClipGroupsPause(const ClipGroupsPause&) = delete;
      ClipGroupsPause& operator=(const ClipGroupsPause&) = delete;
    private:
      const ClipGroups* _previous;
    };

    template <typename F>
    bool for_each_clip_group(dim_t count, F&& f) {
      const ClipGroups* groups = clip_groups();
      if (!groups || count % groups->total != 0)
        return false;
      const ClipGroups copy = *groups;
      const ClipGroupsPause pause;
      const dim_t per_clip = count / copy.total;
      dim_t first = 0;
      for (const dim_t clips : copy.clips) {
        if (clips == 0)
          continue;
        f(first, clips * per_clip);
        first += clips * per_clip;
      }
      return true;
    }

  }
}

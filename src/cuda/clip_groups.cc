#include "cuda/clip_groups.h"

#include <algorithm>
#include <utility>

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    static thread_local const ClipGroups* active = nullptr;

    const ClipGroups* clip_groups() {
      return active;
    }

    // rowinv2 on the L40S (sm_89, cuBLAS 12.9.2, 5.10.2026): Whisper large-v3's decoder self-attention input
    // (3840 x 1280), output and cross-attention query and output (1280 x 1280), first feed-forward (5120 x 1280)
    // and vocabulary (51866 x 1280, and 51872 as the decoder pads it to a multiple of 8) products: every row count
    // 2..320 has the bits of 40-row calls, 3 fills. Not the second feed-forward (1280 x 5120): its split over k
    // changes with the rows (17-27, 28-34, 35-44 rows ...).
    bool rows_independent_product(dim_t m, dim_t n, dim_t k) {
      static const bool sm89 = cublas_verified_on(8, 9);
      if (!sm89 || m < 2 || m > 320 || k != 1280)
        return false;
      return n == 3840 || n == 1280 || n == 5120 || n == 51866 || n == 51872;
    }

    ClipGroups make_clip_groups(const std::vector<dim_t>& batch_offset, dim_t group_size) {
      ClipGroups groups;
      if (group_size <= 0 || batch_offset.empty())
        return groups;
      const dim_t last = *std::max_element(batch_offset.begin(), batch_offset.end());
      groups.clips.assign(last / group_size + 1, 0);
      for (const dim_t index : batch_offset)
        ++groups.clips[index / group_size];
      groups.total = static_cast<dim_t>(batch_offset.size());
      return groups;
    }

    ClipGroupsScope::ClipGroupsScope(ClipGroups groups)
      : _groups(std::move(groups))
      , _previous(active) {
      const auto with_clips = std::count_if(_groups.clips.begin(), _groups.clips.end(),
                                            [](dim_t clips) { return clips > 0; });
      active = with_clips >= 2 ? &_groups : nullptr;
    }

    ClipGroupsScope::~ClipGroupsScope() {
      active = _previous;
    }

    ClipGroupsPause::ClipGroupsPause()
      : _previous(active) {
      active = nullptr;
    }

    ClipGroupsPause::~ClipGroupsPause() {
      active = _previous;
    }

  }
}

#include "cuda/clip_groups.h"

#include <algorithm>
#include <utility>

namespace ctranslate2 {
  namespace cuda {

    static thread_local const ClipGroups* active = nullptr;

    const ClipGroups* clip_groups() {
      return active;
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

#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    // cache [old_rows, heads, time, d] <- rows cache[order[r]], then fresh [rows, heads, t, d]
    // appended along time: Gather(order) + Concat(2) in one CUDA pass where the layout allows
    // (cuda/cache_reorder.h), the two ops otherwise. Same values either way.
    void reorder_and_append(StorageView& cache, const StorageView& order, const StorageView& fresh);

  }
}

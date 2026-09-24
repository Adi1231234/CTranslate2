#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    // keys/values caches [old_rows, heads, time, d] <- rows cache[order[r]], then the fresh keys/values
    // [rows, heads, t, d] appended along time: Gather(order) + Concat(2) for both in one CUDA launch
    // where the layout allows (cuda/cache_reorder.h), the two ops otherwise. Same values either way.
    void reorder_and_append(StorageView& keys, StorageView& values, const StorageView& order,
                            const StorageView& fresh_keys, const StorageView& fresh_values);

  }
}

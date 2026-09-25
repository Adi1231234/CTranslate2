#pragma once

#include <vector>

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace cuda {

    // How many weights a decoding step prefetches into L2 after its self-attention input projection
    // (CT2_L2_PREFETCH, default 0: none). See l2_prefetch().
    int l2_prefetch_count();

    // Asks the GPU to bring buffers into L2 without reading them into registers: one small kernel issues
    // prefetch.global.L2 for each 128-byte line and ends; the lines arrive while the next kernels run.
    // Only a cache hint, so no value anywhere changes. Launched where the decoder's next kernels leave the
    // memory idle (the self-attention's small kernels), for the weights of the Dense layers that follow.
    void l2_prefetch(const std::vector<const StorageView*>& buffers);

  }
}

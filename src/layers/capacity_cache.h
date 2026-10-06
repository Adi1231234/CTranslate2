#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    // CT2_CAPACITY_CACHES=1: a greedy search's self-attention caches of a fixed capacity along time on CUDA (fp16),
    // instead of caches copied whole into new buffers at every step (Concat: on long recordings, whose ~227-token
    // prompts make t ~230-448, a ladder's 25 rows copied ~2.5 GB a step and held two copies at once, which ran the
    // L40S out of memory next to the stream: long11). At its first step a layer moves its cache [rows, heads, t,
    // depth] into [rows, heads, t + steps, depth]; each step writes its keys and values at position `time` and runs
    // the attention products over the first time + 1 positions with the capacity as their batch stride, which gives
    // cuBLAS's bits of the contiguous caches (kernels/cache_stride_check.cu). Finished rows leave by the state's
    // gathers as before. Every row of a greedy search has the same length, so one time serves all layers and rows.
    struct CapacityCaches {
      dim_t steps = 0;                       // the search's steps: the room after the prompt
      dim_t time = -1;                       // positions cached before the step; -1 before the first
      dim_t shared = 0;                      // positions [0, shared) alike in every row (the prompt, repeated)
      void advance() {                       // after each decoder step
        if (time >= 0)
          ++time;
      }
    };

    bool capacity_caches_enabled();
    CapacityCaches* capacity_caches();       // the thread's search's, or null

    class CapacityCacheScope {
    public:
      explicit CapacityCacheScope(CapacityCaches* caches);
      ~CapacityCacheScope();
      CapacityCacheScope(const CapacityCacheScope&) = delete;
      CapacityCacheScope& operator=(const CapacityCacheScope&) = delete;
    private:
      CapacityCaches* const _previous;
    };

    // A decoding step's self-attention on the caches (dot_product_attention's MatMul, SoftMax and MatMul): queries,
    // keys and values [rows, heads, 1, depth]; context [rows, heads, 1, depth], the heads not combined.
    void capacity_attention(const StorageView& queries, const StorageView& keys, const StorageView& values,
                            float scale, StorageView& cached_keys, StorageView& cached_values,
                            StorageView& context);

  }
}

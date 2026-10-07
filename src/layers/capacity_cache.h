#pragma once

#include <utility>
#include <vector>

#include "ctranslate2/storage_view.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/slot_attention.h"
#endif

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

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
    // The slot attention's entries on the device, as int32 words (empty for none).
    StorageView slot_table(const std::vector<cuda::SlotAttention>& entries);
    // The step's keys and values ([rows, heads, 1, depth]) at position `time` of the caches, after each layer's move
    // to the capacity at the search's first step.
    void capacity_append(CapacityCaches& caches, StorageView& cached_keys, StorageView& cached_values,
                         const StorageView& keys, const StorageView& values);
    // The slot attention's entry (cuda/slot_attention.h) for a group of `count` rows from `first` of a search's caches,
    // its scores at `scores`, its queries and outputs from row `row_begin` of the launch's.
    cuda::SlotAttention capacity_group(const CapacityCaches& caches, StorageView& cached_keys,
                                       StorageView& cached_values, void* scores, dim_t first, dim_t count,
                                       dim_t row_begin);
#endif

    // A decoding step's self-attention on the caches (dot_product_attention's MatMul, SoftMax and MatMul): queries,
    // keys and values [rows, heads, 1, depth]; context [rows, heads, 1, depth], the heads not combined.
    void capacity_attention(const StorageView& queries, const StorageView& keys, const StorageView& values,
                            float scale, StorageView& cached_keys, StorageView& cached_values,
                            StorageView& context);

    // A greedy search's rows of a joint step (capacity_parts.cc): its caches this layer, its rows in the step's
    // queries, keys, values and context, and its groups (first row, rows; each one cuBLAS call of its own).
    struct CapacityPart {
      CapacityCaches* caches;
      StorageView* cached_keys;
      StorageView* cached_values;
      dim_t row_begin;
      dim_t rows;
      std::vector<std::pair<dim_t, dim_t>> groups;
    };
    // capacity_attention of every part at once: one launch per product for all their groups of the slot attention,
    // one softmax for all their scores; each row's values those of capacity_attention for its search alone.
    void capacity_attention_parts(const std::vector<CapacityPart>& parts, const StorageView& queries,
                                  StorageView& keys, StorageView& values, float scale, StorageView& context);

  }
}

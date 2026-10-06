#pragma once

#include <vector>

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    struct JointStep;

    // CT2_JOINT_SLOTS=1: a joint decoding part's self-attention caches in slots that stay in place, instead of caches
    // copied whole into new buffers at every step (the beams' Gather and the step's Concat; long5: 15% of the GPU's
    // time on long recordings, whose ~200-token prompts make t ~200-400). A row keeps its parent's slot when it is
    // the parent's first child; another child takes a slot no row keeps and gets a copy of the parent's positions,
    // the only copy left. The attention runs over the slots in slot order, the queries permuted in and the outputs
    // back: cuBLAS gives every entry the bits of the contiguous caches with the slots' stride and the entries in
    // another order (kernels/cache_stride_check.cu: every t 1..448, the L40S). For parts of one input (the
    // stream's batches of one window), whose rows never shrink while the part decodes; from the step after the
    // beams' expansion (rows > inputs).
    struct SlotCache {
      static constexpr dim_t capacity = 448;           // the Whisper decoder's positions
      dim_t time = -1;                                 // positions cached before the step; -1 before its first
      dim_t rows = 0;
      std::vector<StorageView> keys, values;           // per layer, [rows, heads, capacity, depth]
      StorageView maps{DataType::INT32};               // slot_of_row, row_of_slot, fork_src, fork_dst, fork_count
    };

    bool joint_slots();

    // Before a joint step's layers: each slot part's caches made (its first slot step), its plan for the step
    // (cuda/slot_cache.h) and the step's descriptor tables; parts not yet expanded keep the copies this step.
    void prepare_slots(JointStep& joint);
    // After the step's layers.
    void finish_slots(JointStep& joint);

    // A layer's self-attention for the slot parts: the forks' copies and the step's keys and values into the slots,
    // the queries into slot order, each part's scores and values products over its slots, the outputs back.
    void slot_append(const JointStep& joint, const StorageView& keys, const StorageView& values);
    void slot_queries(const JointStep& joint, const StorageView& queries, StorageView& slot_queries);
    void slot_scores(const JointStep& joint, size_t part, const StorageView& slot_queries, float scale,
                     StorageView& scores);
    void slot_values(const JointStep& joint, size_t part, const StorageView& probs, StorageView& slot_context);
    void slot_context(const JointStep& joint, const StorageView& slot_context, StorageView& context);
    dim_t slot_time(const JointStep& joint, size_t part);   // the keys a slot part's step attends to

  }
}

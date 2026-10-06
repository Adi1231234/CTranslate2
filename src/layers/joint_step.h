#pragma once

#include <memory>
#include <vector>

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    struct CapacityCaches;

    // A greedy search's rows in a joint step (GreedySearchRun::joint_rows): what its own decoder call runs under
    // (GreedySearch::search), so that the attention of its rows runs as there: its groups (cuda/clip_groups.h: the
    // rows of each group still decoding, 0 for a finished group), each row's input among its memory entries
    // (cuda/shared_memory_rows.h) and its self-attention caches of a fixed capacity (capacity_cache.h).
    struct SampledRows {
      std::vector<dim_t> group_rows;
      dim_t rows = 0;
      const int32_t* row_input = nullptr;                  // on the device
      dim_t inputs = 0;                                    // its memory entries
      CapacityCaches* capacity = nullptr;
    };

    // One decoder step for the rows of several independent beam searches (TransformerDecoder::decode_joint),
    // while a JointStepScope is active on the thread. The parts' rows are concatenated in part order; each part
    // keeps its own position, self-attention caches and memory keys and values, so the attention layers handle
    // each part's own (MultiHeadAttention::joint_attention), and every other op runs once on all the rows.
    struct JointStep {
      struct Part {
        dim_t row_begin = 0;
        dim_t rows = 0;
        dim_t clips = 0;                                   // inputs still decoding: rows / clips beams each
        std::vector<StorageView*> self_keys;               // its self-attention caches, by layer
        std::vector<StorageView*> self_values;
        std::unique_ptr<StorageView> cache_reorder;        // the beam order its last update_state left, or null
        struct SlotCache* slots = nullptr;                 // its caches in slots this step (slot_cache.h), or null
        dim_t scores_offset = -1;                          // its scores in slot_scores (slot_attention.h), or -1
        const SampledRows* sampled = nullptr;              // a greedy search's rows (after every beam part), or null
        std::vector<StorageView*> memory_keys;             // a greedy part's memory keys and values, by layer
        std::vector<StorageView*> memory_values;
      };
      std::vector<Part> parts;
      dim_t clips = 0;
      size_t layer = 0;                                    // the decoder layer running
      // The slot parts' descriptor tables for the step (slot_cache.cc), on the device as int32 words: a SlotAppend
      // per layer and part, a SlotPermute per part for the queries (row_of_slot) and the outputs (slot_of_row).
      StorageView slot_appends{DataType::INT32};
      StorageView slot_queries{DataType::INT32};
      StorageView slot_outputs{DataType::INT32};
      int slot_parts = 0, slot_max_rows = 0, slot_max_time = 0;
      // The slot parts' self-attention in one launch per product (layers/slot_attention.h): a cuda::SlotAttention
      // per layer and fused part (slot_fused of them: the parts with a scores_offset), and their scores, reused by
      // each layer.
      StorageView slot_attention{DataType::INT32};
      StorageView slot_scores;
      int slot_fused = 0;
      // On the device, per layer: each clip's memory keys and values, [heads][1500][64] from these two pointers.
      StorageView memory_table{DataType::INT32};

      const void* const* memory(size_t layer) const;
    };

    const JointStep* joint_step();

    class JointStepScope {
    public:
      explicit JointStepScope(const JointStep& step);
      ~JointStepScope();
      JointStepScope(const JointStepScope&) = delete;
      JointStepScope& operator=(const JointStepScope&) = delete;
    private:
      const JointStep* _previous;
    };

    // A view of rows [begin, begin + count) of x (along its first dimension).
    StorageView rows_view(StorageView& x, dim_t begin, dim_t count);

  }
}

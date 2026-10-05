#pragma once

#include <memory>
#include <vector>

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

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
      };
      std::vector<Part> parts;
      dim_t clips = 0;
      size_t layer = 0;                                    // the decoder layer running
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

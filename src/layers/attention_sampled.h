#pragma once

#include "ctranslate2/storage_view.h"
#include "joint_step.h"

namespace ctranslate2 {
  namespace layers {

    // A greedy search's part of a joint decoding step (joint_step.h: SampledRows), its attention run as its search's
    // own decoder step runs it (GreedySearch::search: its clip groups, shared memory rows and capacity caches
    // active), on its rows; each writes the part's context in its rows of `context` ([rows, heads, 1, depth]).

    // Self-attention: MultiHeadAttention's capacity path on the part's rows of the step's head-split queries, keys
    // and values (split_heads_with_bias of every row's projection: the same values row by row).
    void sampled_self_attention(const JointStep& joint, const JointStep::Part& part, StorageView& queries,
                                StorageView& keys, StorageView& values, float scale, StorageView& context);

    // Cross-attention: process_cross_attention's split of the part's rows of the queries' projection with their bias,
    // then dot_product_attention against the part's memory keys and values (its shared memory rows).
    void sampled_cross_attention(const JointStep& joint, const JointStep::Part& part, StorageView& proj,
                                 const StorageView* bias, dim_t heads, float scale, StorageView& context);

  }
}

#pragma once

#include "ctranslate2/layers/attention.h"

namespace ctranslate2 {
  namespace layers {

    // MultiHeadAttention's attention between its projected, head-split queries, keys and values (attention.cc),
    // also run per part by joint_attention (attention_joint.cc). True when output holds the heads combined
    // ([batch, time, heads, depth] under the queries' shape).
    bool dot_product_attention(const StorageView& queries,
                               const StorageView& keys,
                               const StorageView& values,
                               const StorageView* values_lengths,
                               const StorageView* relative_position_keys,
                               const StorageView* relative_asymmetric_position_keys,
                               const StorageView* relative_position_values,
                               const StorageView* relative_attention_bias,
                               dim_t relative_left_max_position,
                               dim_t relative_right_max_position,
                               dim_t maximum_relative_position,
                               StorageView& output,
                               StorageView* attention,
                               bool return_normalized_attention,
                               float queries_scale,
                               bool is_decoder,
                               bool with_cache,
                               dim_t beam_size,
                               Alibi* alibi,
                               StorageView* position_bias);

  }
}

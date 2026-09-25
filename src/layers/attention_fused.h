#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    // Attention o = SoftMax(MatMul(queries, keys^T, scale)) values in one CUDA pass where that equals the three
    // ops bit for bit (cuda/exact_attention.h): [batch, heads, time, depth], fp16, no lengths. The output keeps the
    // queries' shape but holds the heads combined, [batch, time, heads, depth] (combine_heads' layout).
    bool attention_fusable(const StorageView& queries, const StorageView& keys, const StorageView& values);
    void attention_fused(const StorageView& queries, const StorageView& keys, const StorageView& values,
                         float scale, StorageView& output);

    // The same straight from the self-attention's fused projection proj [batch, time, 3 * heads * depth] without
    // its bias, and that bias (split_heads_fused.h's inputs): the bias add and head split happen as the kernels
    // read, so queries, keys and values are never written out. output: [batch, heads, time, depth] holding the
    // heads combined.
    bool attention_qkv_fusable(const StorageView& proj, const StorageView* bias, dim_t heads, dim_t depth);
    void attention_qkv_fused(const StorageView& proj, const StorageView* bias, dim_t heads, float scale,
                             StorageView& output);

  }
}

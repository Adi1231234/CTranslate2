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

  }
}

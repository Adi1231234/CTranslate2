#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    // Attention scores and their softmax in one CUDA pass where that equals MatMul(trans_b, scale) + SoftMax
    // bit for bit (cuda/scores_softmax.h): queries and keys [batch, heads, time, depth], fp16, no lengths.
    bool scores_softmax_fusable(const StorageView& queries, const StorageView& keys);
    void scores_softmax_fused(const StorageView& queries, const StorageView& keys, float scale,
                              StorageView& attention);

  }
}

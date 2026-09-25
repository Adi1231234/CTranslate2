#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    // Whisper decoder cross-attention on cuda/cross_attention.h's single kernel. The cache is made in its fragment
    // order when created ([batch, heads, 1504, 64] instead of the keys and values [batch, heads, 1500, 64]), for
    // a decoding that asks for no attention weights and no memory lengths; every later step then runs the kernel.

    // At the cache's creation: whether keys [batch, heads, time, depth] can go to the fragment order.
    bool cross_fragments_apply(const StorageView& keys);
    // Replaces keys and values by their fragment order.
    void to_cross_fragments(StorageView& keys, StorageView& values);
    // Whether a cache was made by to_cross_fragments (memory_time: the encoder output's length).
    bool are_cross_fragments(const StorageView& keys, dim_t memory_time);
    // queries [batch, heads, m, 64] -> output [batch, m, heads, 64] in the queries' shape (heads combined).
    void cross_attention_fused(const StorageView& queries, const StorageView& kf, const StorageView& vf,
                               float scale, StorageView& output);

    // CT2_CROSS_CHECK=1: caches keep the plain order, and every cross-attention the three ops compute is also
    // computed by the kernel and compared bit for bit (on the host; mismatches are printed to stderr).
    bool cross_check_enabled();
    void cross_check(const StorageView& queries, const StorageView& keys, const StorageView& values, float scale,
                     const StorageView& reference);

  }
}

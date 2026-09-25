#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    // Whisper decoder cross-attention on cuda/cross_attention.h's single kernel, where it equals the three ops bit
    // for bit: queries [batch, heads, m, 64], keys and values [batch, heads, 1500, 64], fp16, no lengths. The output
    // keeps the queries' shape but holds the heads combined, [batch, m, heads, 64] (combine_heads' layout).
    // residue: set to the key tile residue the kernel must use.
    bool cross_attention_fusable(const StorageView& queries, const StorageView& keys, const StorageView& values,
                                 int& residue);
    void cross_attention_fused(const StorageView& queries, const StorageView& keys, const StorageView& values,
                               float scale, int residue, StorageView& output);

    // CT2_CROSS_CHECK=1: the three ops run, and the kernel's result is compared with theirs bit for bit (on the
    // host; mismatches are printed to stderr, and the totals at exit).
    bool cross_check_enabled();
    void cross_check(const StorageView& queries, const StorageView& keys, const StorageView& values, float scale,
                     const StorageView& reference);

  }
}

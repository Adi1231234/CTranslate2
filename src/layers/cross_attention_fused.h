#pragma once

#include "ctranslate2/layers/common.h"
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

    // The decoding steps (the cache made): the queries' Dense layer and head split too, from the normed decoder
    // state x [rows, 1, d]. m: set to the queries per clip (rows / cached clips); output as above, [clips, heads,
    // m, 64] holding [clips, m, heads, 64].
    bool cross_attention_q_fusable(const StorageView& x, const Dense& linear, const StorageView& keys,
                                   const StorageView& values, dim_t& m, int& residue);
    void cross_attention_fused_q(const StorageView& x, const Dense& linear, const StorageView& keys,
                                 const StorageView& values, float scale, dim_t m, int residue, StorageView& output);

    // The two conditions above without CT2_CROSS_CHECK (for the check): the key tile residue or -1; whether the
    // queries' Dense layer goes in the kernel too.
    int cross_kernel_residue(const StorageView& queries, const StorageView& keys, const StorageView& values);
    bool cross_q_kernel_applies(const StorageView& x, const Dense& linear, const StorageView& keys,
                                const StorageView& values, dim_t& m, int& residue);

    // CT2_CROSS_CHECK=1 (cross_attention_check.cc): the ops run, and the kernel's results, without and with the
    // queries' Dense layer, are compared with theirs bit for bit (on the host; mismatches are printed to stderr,
    // and the totals at exit). reference: the ops' attention output [clips, heads, m, 64].
    bool cross_check_enabled();
    void cross_check(const StorageView& queries, const StorageView& keys, const StorageView& values, float scale,
                     const StorageView& reference);
    void cross_check_q(const StorageView& x, const Dense& linear, const StorageView& keys, const StorageView& values,
                       float scale, const StorageView& reference);

  }
}

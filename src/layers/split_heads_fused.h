#pragma once

#include <initializer_list>

#include "ctranslate2/layers/common.h"
#include "ctranslate2/padder.h"

namespace ctranslate2 {
  namespace layers {

    // Dense + MultiHeadAttention::split_heads + ops::Split, fused on the GPU: the caller runs
    // linear.compute_without_bias(x, proj) and split_heads_with_bias(proj, linear.bias(), ...).
    // Applies to fp16 CUDA inputs of rank 3 without padding, when the linear can defer its bias
    // (and never with CT2_CUDA_STOCK_KERNELS=1). The output bits equal the unfused path's.
    bool split_heads_fusable(const StorageView& x, const Dense& linear, const Padder* padder,
                             dim_t d_head);

    // proj: [batch, time, parts * heads * d_head] -> each out: [rows, heads, t, d_head], part by
    // part. With beam_size > 1 (single-step queries), the beams of a batch become its time axis,
    // as in split_heads.
    void split_heads_with_bias(const StorageView& proj,
                               const StorageView* bias,
                               std::initializer_list<StorageView*> outs,
                               dim_t heads,
                               dim_t beam_size = 1);

  }
}

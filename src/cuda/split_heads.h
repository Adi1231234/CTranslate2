#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // Attention input layout in one pass. x is [rows, time, parts * heads * head_dim] (a fused
    // projection without its bias); out[p] is [rows, heads, time, head_dim] and receives part p:
    //   out[p][r, h, t, i] = x[r, t, (p * heads + h) * head_dim + i] + bias[(p * heads + h) * head_dim + i]
    // with the bias add of Dense (cuda::plus<__half>: float(bias) + float(x), rounded once to half),
    // so the result equals Dense + MultiHeadAttention::split_heads + ops::Split bit for bit.
    // bias may be null. parts <= 3, head_dim a multiple of split_heads_bias_granule, and every
    // pointer split_heads_bias_aligned.
    constexpr dim_t split_heads_bias_granule = 8;
    bool split_heads_bias_aligned(const void* p);   // null counts as aligned
    void split_heads_bias(const float16_t* x, const float16_t* bias, float16_t* const* out, int parts,
                          dim_t rows, dim_t time, dim_t heads, dim_t head_dim);

  }
}

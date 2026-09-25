#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // A sublayer's output and the next sublayer's pre-norm in one kernel: for each of `rows` rows of `depth`
    // values, sum = (bias + x) + residual with the per-element fp16 arithmetic of BiasAdd's residual path
    // (bias_add_vec.cuh), then normed = LayerNorm(sum) with the arithmetic of ops::LayerNorm's CUDA kernel
    // (the same 512-thread block reduction, the same expressions), so both outputs are those two ops' bits.
    // sum may be x (in place).
    void residual_norm(const float16_t* x, const float16_t* bias, const float16_t* residual,
                       const float16_t* gamma, const float16_t* beta, float epsilon,
                       float16_t* sum, float16_t* normed, dim_t rows, dim_t depth);

  }
}

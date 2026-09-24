#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // Whisper encoder self-attention: p = SoftMax(MatMul(q, k^T, alpha)) in one pass, bit for bit equal to the
    // two ops (fp16, no lengths) where that was verified: sm_120 with cuBLAS 12.9.2, 64-dim heads,
    // 1500 x 1500 (ops/attention_scores_softmax.cuh; tools/turing/kernels/qk_hmma_probe.cu and
    // scores_softmax_check.cu). q is [batch, m, 64], k [batch, n, 64], p [batch, m, n].
    bool attention_scores_softmax_applies(dim_t m, dim_t n, dim_t depth, const void* q, const void* k);
    void attention_scores_softmax(const float16_t* q, const float16_t* k, float16_t* p,
                                  dim_t batch, dim_t m, dim_t n, float alpha);

  }
}

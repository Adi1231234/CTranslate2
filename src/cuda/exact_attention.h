#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // Whisper encoder self-attention o = SoftMax(MatMul(q, k^T, alpha)) v in one pass, bit for bit equal to the
    // three ops (fp16, no lengths) where that was verified: sm_120 with cuBLAS 12.9.2, 64-dim heads, 1500 x
    // 1500 (ops/exact_attention.cuh; tools/turing/kernels/exact_attention_check.cu). q is [batch, m, 64],
    // k and v [batch, n, 64], o [batch, m, 64]; workspace holds exact_attention_workspace_bytes.
    bool exact_attention_applies(dim_t m, dim_t n, dim_t depth, const void* q, const void* k, const void* v);
    size_t exact_attention_workspace_bytes(dim_t batch, dim_t n);
    void exact_attention(const float16_t* q, const float16_t* k, const float16_t* v, void* workspace,
                         float16_t* o, dim_t batch, dim_t m, dim_t n, float alpha);

  }
}

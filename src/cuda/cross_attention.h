#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // Whisper decoder cross-attention in one kernel (ops/cross_attention.cuh), bit for bit the MatMul, SoftMax and
    // MatMul it replaces, where their cuBLAS kernels were matched: sm_120 with cuBLAS 12.9.2, fp16, 64-dim heads,
    // 1500 encoder positions. CT2_CROSS_ATTN=0 keeps the three ops.
    bool cross_attention_applies(dim_t keys, dim_t depth);

    // k, v: [entries][1500][64] -> kf, vf: [entries][1504][64] in fragment order (cross_attention_layout.cuh).
    void cross_attention_layout(const float16_t* k, const float16_t* v, float16_t* kf, float16_t* vf,
                                dim_t entries);

    // q: [clips][heads][m][64]; kf, vf as above (entries = clips * heads); o: [clips][m][heads][64].
    void cross_attention(const float16_t* q, const float16_t* kf, const float16_t* vf, float16_t* o,
                         dim_t clips, dim_t heads, dim_t m, float alpha);

  }
}

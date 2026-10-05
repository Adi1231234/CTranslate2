#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // c = a w^T for a Whisper encoder product (fp16, a [m, k], w [n, k], c [m, n], fp32 accumulation) on a pinned
    // cuBLASLt algorithm where one gives the bits of the cuBLAS call CTranslate2 makes with less energy (the batched
    // path is bound by the GPU's power limit, cuBLAS's heuristic picks for time): on sm_89 with cuBLAS 12.9.2, the
    // first feed-forward (5120 x 1280) on algorithm 21, tile 24, 9 stages, swizzled, no split-K: 304 against 379 mJ
    // and 902 against 1080 us at 8 clips (tools/turing/kernels/encoder_algo_search.cu, round35; every other encoder
    // product's exact algorithms take more than cuBLAS's own). Returns false, nothing launched, where none applies;
    // CT2_ENC_LT=0 keeps cuBLAS's pick.
    bool encoder_lt_gemm(const float16_t* a, const float16_t* w, float16_t* c, dim_t m, dim_t n, dim_t k);

  }
}

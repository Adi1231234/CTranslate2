#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // The Whisper decoder's products in tiled_split_gemm.cuh's kernel instead of cuBLAS, with cuBLAS's bits
    // (tools/turing/kernels/decoder_gemm_probe.cu): C = A W^T (fp16, alpha 1, beta 0, contiguous rows) for groups
    // of rows (group_rows, summing to m), each with what a cuBLAS call with that group alone computes. Only on
    // sm_89 with cuBLAS 12.9.2, k = 1280 (the row-independent products, cuda/clip_groups.h: one chain over k, any
    // groups of 2..320 rows in all) or n = 1280, k = 5120 (the second feed-forward: each group's split, 2..48 rows
    // a group). Each kind of product takes the tile CT2_DECODER_TILES names for it, e.g.
    // "qkv=64x16/8,o=64x16/8,ffn1=64x32/8,vocab=64x128/4,ffn2=64x16/8" (rows x columns / pipeline stages; the
    // kinds: qkv 3840 x 1280, o 1280 x 1280, ffn1 5120 x 1280, vocab 51866 or 51872 x 1280, ffn2 1280 x 5120);
    // a kind not named stays with cuBLAS. Returns false, nothing launched, where it does not apply.
    bool decoder_gemm(const float16_t* a, const float16_t* w, float16_t* c, dim_t m, dim_t n, dim_t k,
                      const std::vector<dim_t>& group_rows, cudaStream_t stream);

  }
}

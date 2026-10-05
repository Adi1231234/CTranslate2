#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // C = A W^T (fp16, alpha 1, beta 0; A [m x k], W [n x k], C [m x n] contiguous rows) for consecutive groups
    // of rows (group_rows, summing to m), each with the bits cuBLAS gives a call with that group's rows alone, in
    // one pass over the weights (grouped_split_gemm.cuh). Only where that arithmetic is known: the Whisper
    // decoder's second feed-forward (n 1280, k 5120) on sm_89 with cuBLAS 12.9.2, groups of 2..48 rows, at most
    // 160 rows. Returns false (nothing launched) elsewhere.
    bool grouped_split_gemm(const float16_t* a, const float16_t* w, float16_t* c, dim_t n, dim_t k,
                            const std::vector<dim_t>& group_rows, cudaStream_t stream);

  }
}

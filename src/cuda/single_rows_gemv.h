#pragma once

#include <vector>

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // Rows of a product C = A W^T (fp16, COMPUTE_32F; W n x k), each with the bits of a call of its own of one row
    // (cuBLAS's gemv), in launches that read W once for up to 16 rows (single_rows_gemv.cuh): the decoder's groups of
    // one row (a sampled ladder's temperatures with one hypothesis left) after a joint call. rows: each one's first
    // index in A and C (A's rows of k values, C's of n). False, nothing launched, where the arithmetic is not
    // recovered (another shape, device or cuBLAS) or CT2_SINGLE_ROWS=0.
    bool single_rows_gemv(const void* a, const void* w, void* c, dim_t n, dim_t k, const std::vector<dim_t>& rows);

  }
}

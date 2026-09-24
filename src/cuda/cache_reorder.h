#pragma once

#include <cstdint>

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // One attention cache update of beam search in one pass: the gather that reorders the beams
    // (Decoder::update_state) and the time concatenation of the new step (MultiHeadAttention).
    // cache is [old_rows, heads, time, head_dim], fresh [rows, heads, fresh_time, head_dim], order
    // [rows] (row r takes cache row order[r]); out is [rows, heads, time + fresh_time, head_dim]:
    //   out[r, h, s]        = cache[order[r], h, s]  for s < time
    //   out[r, h, time + s] = fresh[r, h, s]
    // Pure data movement, so the values equal Gather + Concat bit for bit. Needs
    // head_dim * sizeof(T) to be a multiple of 16 and 16-byte aligned buffers.
    bool cache_reorder_supported(const void* cache, const void* fresh, const void* out,
                                 dim_t head_dim, dim_t type_size);
    template <typename T>
    void reorder_append(const T* cache, const int32_t* order, const T* fresh, T* out,
                        dim_t rows, dim_t heads, dim_t time, dim_t fresh_time, dim_t head_dim);

  }
}

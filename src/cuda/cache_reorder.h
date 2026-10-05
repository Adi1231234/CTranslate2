#pragma once

#include <cstdint>

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // Attention cache updates of beam search in one launch: for each of `count` (<= 2, keys and
    // values) caches, the gather that reorders the beams (Decoder::update_state) and the time
    // concatenation of the new step (MultiHeadAttention). cache[c] is [old_rows, heads, time,
    // head_dim], fresh[c] [rows, heads, fresh_time, head_dim], order [rows] (row r takes cache row
    // order[r]); out[c] is [rows, heads, time + fresh_time, head_dim]:
    //   out[r, h, s]        = cache[order[r], h, s]  for s < time
    //   out[r, h, time + s] = fresh[r, h, s]
    // Pure data movement, so the values equal Gather + Concat bit for bit. Needs
    // head_dim * sizeof(T) to be a multiple of 16 and 16-byte aligned buffers.
    bool cache_reorder_supported(const void* p, dim_t head_dim, dim_t type_size);
    template <typename T>
    void reorder_append(const T* const* cache, const T* const* fresh, T* const* out, int count,
                        const int32_t* order, dim_t rows, dim_t heads, dim_t time, dim_t fresh_time,
                        dim_t head_dim);

    // The same for the keys and values of several searches' caches in one launch (layers/joint_step.h), one step
    // appended (fresh_time 1), fp16: part p's caches [rows, heads, time, head_dim] at cache[p], its fresh [rows,
    // heads, 1, head_dim] at fresh[p], out [rows, heads, time + 1, head_dim] at out[p], its beam order (or null:
    // row r keeps row r). At most max_parts parts.
    struct CacheParts {
      static constexpr int max_parts = 16;
      int count = 0;
      const void* cache[max_parts][2];
      const void* fresh[max_parts][2];
      void* out[max_parts][2];
      const int32_t* order[max_parts];
      int rows[max_parts];
      int time[max_parts];
    };
    void reorder_append_parts(const CacheParts& parts, dim_t heads, dim_t head_dim);

  }
}

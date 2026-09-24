#ifndef CT2_USE_HIP

#include "cuda/cache_reorder.h"

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    struct CacheBuffers {
      const uint4* cache[2];
      const uint4* fresh[2];
      uint4* out[2];
    };

    // One thread per 16-byte vector of the outputs, in out order (coalesced writes; each source run
    // of a (row, head) is contiguous too).
    __global__ void reorder_append_kernel(CacheBuffers b, const int32_t* order, unsigned heads,
                                          unsigned time, unsigned fresh_time, unsigned head_vecs,
                                          size_t per_cache, size_t total) {
      size_t v = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
      if (v >= total)
        return;
      const unsigned c = unsigned(v / per_cache);
      v -= c * per_cache;
      const unsigned out_time = time + fresh_time;
      const size_t rh = v / (size_t(out_time) * head_vecs);        // r * heads + h
      const unsigned rest = unsigned(v - rh * out_time * head_vecs);
      const unsigned s = rest / head_vecs, i = rest - s * head_vecs;
      const size_t r = rh / heads, h = rh - r * heads;
      b.out[c][v] = s < time
        ? b.cache[c][((size_t(order[r]) * heads + h) * time + s) * head_vecs + i]
        : b.fresh[c][(rh * fresh_time + (s - time)) * head_vecs + i];
    }

    bool cache_reorder_supported(const void* p, dim_t head_dim, dim_t type_size) {
      return (head_dim * type_size) % 16 == 0 && reinterpret_cast<uintptr_t>(p) % 16 == 0;
    }

    template <typename T>
    void reorder_append(const T* const* cache, const T* const* fresh, T* const* out, int count,
                        const int32_t* order, dim_t rows, dim_t heads, dim_t time, dim_t fresh_time,
                        dim_t head_dim) {
      CacheBuffers b{};
      for (int c = 0; c < count; ++c) {
        b.cache[c] = reinterpret_cast<const uint4*>(cache[c]);
        b.fresh[c] = reinterpret_cast<const uint4*>(fresh[c]);
        b.out[c] = reinterpret_cast<uint4*>(out[c]);
      }
      const unsigned head_vecs = head_dim * sizeof (T) / 16;
      const size_t per_cache = size_t(rows) * heads * (time + fresh_time) * head_vecs;
      const size_t total = per_cache * count;
      if (total == 0)
        return;
      constexpr unsigned threads = 256;
      reorder_append_kernel<<<(total + threads - 1) / threads, threads, 0, get_cuda_stream()>>>(
        b, order, heads, time, fresh_time, head_vecs, per_cache, total);
    }

    template void reorder_append(const float16_t* const*, const float16_t* const*, float16_t* const*,
                                 int, const int32_t*, dim_t, dim_t, dim_t, dim_t, dim_t);

  }
}

#endif

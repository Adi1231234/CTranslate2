#ifndef CT2_USE_HIP

#include "cuda/cache_reorder.h"

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // One thread per 16-byte vector of out, in out order (coalesced writes; each source run of a
    // (row, head) is contiguous too).
    __global__ void reorder_append_kernel(const uint4* cache, const int32_t* order, const uint4* fresh,
                                          uint4* out, unsigned heads, unsigned time, unsigned fresh_time,
                                          unsigned head_vecs, size_t total) {
      const size_t v = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
      if (v >= total)
        return;
      const unsigned out_time = time + fresh_time;
      const size_t rh = v / (size_t(out_time) * head_vecs);        // r * heads + h
      const unsigned rest = unsigned(v - rh * out_time * head_vecs);
      const unsigned s = rest / head_vecs, i = rest - s * head_vecs;
      const size_t r = rh / heads, h = rh - r * heads;
      out[v] = s < time
        ? cache[((size_t(order[r]) * heads + h) * time + s) * head_vecs + i]
        : fresh[(rh * fresh_time + (s - time)) * head_vecs + i];
    }

    bool cache_reorder_supported(const void* cache, const void* fresh, const void* out,
                                 dim_t head_dim, dim_t type_size) {
      auto aligned = [](const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; };
      return (head_dim * type_size) % 16 == 0 && aligned(cache) && aligned(fresh) && aligned(out);
    }

    template <typename T>
    void reorder_append(const T* cache, const int32_t* order, const T* fresh, T* out,
                        dim_t rows, dim_t heads, dim_t time, dim_t fresh_time, dim_t head_dim) {
      const unsigned head_vecs = head_dim * sizeof (T) / 16;
      const size_t total = size_t(rows) * heads * (time + fresh_time) * head_vecs;
      if (total == 0)
        return;
      constexpr unsigned threads = 256;
      reorder_append_kernel<<<(total + threads - 1) / threads, threads, 0, get_cuda_stream()>>>(
        reinterpret_cast<const uint4*>(cache), order, reinterpret_cast<const uint4*>(fresh),
        reinterpret_cast<uint4*>(out), heads, time, fresh_time, head_vecs, total);
    }

    template void reorder_append(const float16_t*, const int32_t*, const float16_t*, float16_t*,
                                 dim_t, dim_t, dim_t, dim_t, dim_t);

  }
}

#endif

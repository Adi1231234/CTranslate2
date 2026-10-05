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

    // reorder_append_kernel for several parts, a block per head row of a part's output (keys, then values, in part
    // order): the parent's head row is one contiguous run of time x head_vecs vectors in the cache and in the
    // output, so the threads copy it with no index arithmetic per vector, then the step's vectors. The block's part
    // is found from its first head row (PartRows), its fields picked in loops over constant indices: a parameter
    // array indexed by a computed part would be copied to local memory by every thread.
    struct PartRows {
      unsigned end[CacheParts::max_parts];                  // each part's head rows (keys and values) end here
    };

    __global__ void reorder_append_parts_kernel(CacheParts parts, PartRows ranges, unsigned heads,
                                                unsigned head_vecs) {
      const unsigned block = blockIdx.x;
      unsigned begin = 0, time = 0, rows = 0;
      const void* cache[2] = {};
      const void* fresh[2] = {};
      void* out[2] = {};
      const int32_t* order = nullptr;
      #pragma unroll
      for (int q = 0; q < CacheParts::max_parts; ++q)
        if (q < parts.count && block < ranges.end[q] && (q == 0 || block >= ranges.end[q - 1])) {
          begin = q == 0 ? 0 : ranges.end[q - 1];
          time = unsigned(parts.time[q]);
          rows = unsigned(parts.rows[q]);
          cache[0] = parts.cache[q][0]; cache[1] = parts.cache[q][1];
          fresh[0] = parts.fresh[q][0]; fresh[1] = parts.fresh[q][1];
          out[0] = parts.out[q][0]; out[1] = parts.out[q][1];
          order = parts.order[q];
        }
      unsigned rh = block - begin;                            // c * rows * heads + r * heads + h
      const unsigned c = rh >= rows * heads;                  // keys, then values
      rh -= c * rows * heads;
      const unsigned r = rh / heads, h = rh - r * heads;
      const size_t from = order ? size_t(order[r]) * heads + h : rh;
      const unsigned run = time * head_vecs;
      const uint4* src = static_cast<const uint4*>(c ? cache[1] : cache[0]) + from * run;
      uint4* dst = static_cast<uint4*>(c ? out[1] : out[0]) + size_t(rh) * (run + head_vecs);
      for (unsigned v = threadIdx.x; v < run; v += blockDim.x)
        dst[v] = src[v];
      if (threadIdx.x < head_vecs)
        dst[run + threadIdx.x] = static_cast<const uint4*>(c ? fresh[1] : fresh[0])[size_t(rh) * head_vecs
                                                                                   + threadIdx.x];
    }

    void reorder_append_parts(const CacheParts& parts, dim_t heads, dim_t head_dim) {
      const unsigned head_vecs = head_dim * sizeof (float16_t) / 16;
      PartRows ranges{};
      unsigned blocks = 0;
      for (int p = 0; p < parts.count; ++p) {
        blocks += 2 * unsigned(parts.rows[p]) * unsigned(heads);
        ranges.end[p] = blocks;
      }
      if (blocks == 0)
        return;
      reorder_append_parts_kernel<<<blocks, 64, 0, get_cuda_stream()>>>(parts, ranges, unsigned(heads),
                                                                        head_vecs);
    }

  }
}

#endif

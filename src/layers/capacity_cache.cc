#include "capacity_cache.h"

#include <stdexcept>

#include "ctranslate2/ops/ops.h"
#include "ctranslate2/primitives.h"
#include "env.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/utils.h"
#endif

namespace ctranslate2 {
  namespace layers {

    static thread_local CapacityCaches* active = nullptr;

    bool capacity_caches_enabled() {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      static const bool on = read_bool_from_env("CT2_CAPACITY_CACHES");
      return on;
#else
      return false;
#endif
    }

    CapacityCaches* capacity_caches() {
      return active;
    }

    CapacityCacheScope::CapacityCacheScope(CapacityCaches* caches)
      : _previous(active) {
      active = caches;
    }

    CapacityCacheScope::~CapacityCacheScope() {
      active = _previous;
    }

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
    // dst[:, :, at:at + src_time, :] = src for [rows * heads] blocks of time x depth fp16 values.
    static void copy_positions(const StorageView& src, StorageView& dst, dim_t at) {
      const size_t depth_bytes = src.dim(3) * src.item_size();
      CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(dst.buffer()) + at * depth_bytes, dst.dim(2) * depth_bytes,
                                   src.buffer(), src.dim(2) * depth_bytes, src.dim(2) * depth_bytes,
                                   src.dim(0) * src.dim(1), cudaMemcpyDeviceToDevice, cuda::get_cuda_stream()));
    }

    // The cache in a buffer of `capacity` positions (its first step).
    static void to_capacity(StorageView& cache, dim_t capacity) {
      StorageView moved({cache.dim(0), cache.dim(1), capacity, cache.dim(3)}, cache.dtype(), cache.device());
      copy_positions(cache, moved, 0);
      cache = std::move(moved);
    }
#endif

    void capacity_attention(const StorageView& queries, const StorageView& keys, const StorageView& values,
                            float scale, StorageView& cached_keys, StorageView& cached_values,
                            StorageView& context) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      CapacityCaches& caches = *active;
      if (caches.time < 0 || cached_keys.dim(2) == caches.time) {   // the search's first step: each layer moves
        caches.time = cached_keys.dim(2);
        to_capacity(cached_keys, caches.time + caches.steps);
        to_capacity(cached_values, caches.time + caches.steps);
      }
      const dim_t capacity = cached_keys.dim(2), time = caches.time;
      if (time >= capacity || cached_values.dim(2) != capacity)
        throw std::logic_error("A capacity cache has no room for the step");
      copy_positions(keys, cached_keys, time);
      copy_positions(values, cached_values, time);

      const dim_t rows = queries.dim(0), heads = queries.dim(1), depth = queries.dim(3), t = time + 1;
      StorageView scores({rows, heads, 1, t}, queries.dtype(), queries.device());
      primitives<Device::CUDA>::gemm_batch_strided(false, true, 1, t, depth, scale,
                                                   queries.data<float16_t>(), depth, depth,
                                                   cached_keys.data<float16_t>(), depth, capacity * depth,
                                                   0.f, scores.data<float16_t>(), t, t, rows * heads);
      ops::SoftMax()(scores, nullptr, scores);
      context.resize({rows, heads, 1, depth});
      primitives<Device::CUDA>::gemm_batch_strided(false, false, 1, depth, t, 1.f,
                                                   scores.data<float16_t>(), t, t,
                                                   cached_values.data<float16_t>(), depth, capacity * depth,
                                                   0.f, context.data<float16_t>(), depth, depth, rows * heads);
#else
      (void)queries; (void)keys; (void)values; (void)scale; (void)cached_keys; (void)cached_values; (void)context;
      throw std::logic_error("Capacity caches need CUDA");
#endif
    }

  }
}

#include "capacity_cache.h"

#include <stdexcept>

#include "ctranslate2/ops/ops.h"
#include "ctranslate2/primitives.h"
#include "env.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include <cstring>
#  include <utility>
#  include <vector>
#  include "cuda/cache_pool.h"
#  include "cuda/clip_groups.h"
#  include "cuda/slot_attention.h"
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
      const cuda::CachePoolScope long_lived;                 // the search's caches (cuda/cache_pool.h)
      StorageView moved({cache.dim(0), cache.dim(1), capacity, cache.dim(3)}, cache.dtype(), cache.device());
      copy_positions(cache, moved, 0);
      cache = std::move(moved);
    }
#endif

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
    void capacity_append(CapacityCaches& caches, StorageView& cached_keys, StorageView& cached_values,
                         const StorageView& keys, const StorageView& values) {
      if (caches.time < 0 || cached_keys.dim(2) == caches.time) {   // the search's first step: each layer moves
        caches.time = cached_keys.dim(2);
        caches.shared = caches.time;                         // every row's caches the prompt's, repeated
        to_capacity(cached_keys, caches.time + caches.steps);
        to_capacity(cached_values, caches.time + caches.steps);
      }
      if (caches.time >= cached_keys.dim(2) || cached_values.dim(2) != cached_keys.dim(2))
        throw std::logic_error("A capacity cache has no room for the step");
      copy_positions(keys, cached_keys, caches.time);
      copy_positions(values, cached_values, caches.time);
    }

#endif

    void capacity_attention(const StorageView& queries, const StorageView& keys, const StorageView& values,
                            float scale, StorageView& cached_keys, StorageView& cached_values,
                            StorageView& context) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      CapacityCaches& caches = *active;
      capacity_append(caches, cached_keys, cached_values, keys, values);
      const dim_t capacity = cached_keys.dim(2), time = caches.time;
      const dim_t rows = queries.dim(0), heads = queries.dim(1), depth = queries.dim(3), t = time + 1;
      StorageView scores({rows, heads, 1, t}, queries.dtype(), queries.device());
      context.resize({rows, heads, 1, depth});
      // Each group of rows (cuda/clip_groups.h) as gemm_batch_strided splits a call: a group of 5 rows from 32
      // positions in slot_attention.cuh's kernels, one launch for them all, the prompt's positions, alike in every
      // row, read once (CT2_SLOT_ATTENTION); the others their own cuBLAS calls.
      std::vector<std::pair<dim_t, dim_t>> groups;           // first row, rows
      if (!cuda::for_each_clip_group(rows, [&](dim_t first, dim_t count) { groups.emplace_back(first, count); }))
        groups.emplace_back(0, rows);
      const float16_t* q = queries.data<float16_t>();
      float16_t* k = cached_keys.data<float16_t>();
      float16_t* v = cached_values.data<float16_t>();
      float16_t* p = scores.data<float16_t>();
      float16_t* out = context.data<float16_t>();
      const dim_t row_cache = heads * capacity * depth;
      std::vector<cuda::SlotAttention> fused;
      std::vector<std::pair<dim_t, dim_t>> own;
      for (const auto& [first, count] : groups) {
        const int ti = static_cast<int>(t);
        if (cuda::slot_attention_enabled() && cuda::slot_attention_applies(count, heads, depth, ti))
          fused.push_back(capacity_group(caches, cached_keys, cached_values, p + first * heads * t, first, count,
                                         first));
        else
          own.emplace_back(first, count);
      }
      const StorageView table = slot_table(fused);
      const auto* parts = reinterpret_cast<const cuda::SlotAttention*>(table.data<int32_t>());
      const cuda::ClipGroupsPause alone;                     // each own group one call
      if (!fused.empty())
        cuda::slot_attention_scores(parts, static_cast<int>(fused.size()), q, static_cast<int>(heads),
                                    static_cast<int>(t), scale);
      for (const auto& [first, count] : own)
        primitives<Device::CUDA>::gemm_batch_strided(false, true, 1, t, depth, scale,
                                                     q + first * heads * depth, depth, depth,
                                                     k + first * row_cache, depth, capacity * depth,
                                                     0.f, p + first * heads * t, t, t, count * heads);
      ops::SoftMax()(scores, nullptr, scores);
      if (!fused.empty())
        cuda::slot_attention_output(parts, static_cast<int>(fused.size()), out, static_cast<int>(heads));
      for (const auto& [first, count] : own)
        primitives<Device::CUDA>::gemm_batch_strided(false, false, 1, depth, t, 1.f,
                                                     p + first * heads * t, t, t,
                                                     v + first * row_cache, depth, capacity * depth,
                                                     0.f, out + first * heads * depth, depth, depth, count * heads);
#else
      (void)queries; (void)keys; (void)values; (void)scale; (void)cached_keys; (void)cached_values; (void)context;
      throw std::logic_error("Capacity caches need CUDA");
#endif
    }

  }
}

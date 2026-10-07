#include "capacity_cache.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "ctranslate2/primitives.h"
#include "joint_parts.h"
#include "joint_step.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/clip_groups.h"
#endif

namespace ctranslate2 {
  namespace layers {

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
    StorageView slot_table(const std::vector<cuda::SlotAttention>& entries) {
      StorageView table(DataType::INT32);
      if (!entries.empty()) {
        std::vector<int32_t> words(entries.size() * sizeof (cuda::SlotAttention) / sizeof (int32_t));
        std::memcpy(words.data(), entries.data(), entries.size() * sizeof (cuda::SlotAttention));
        table = StorageView({static_cast<dim_t>(words.size())}, words).to(Device::CUDA);
      }
      return table;
    }

    cuda::SlotAttention capacity_group(const CapacityCaches& caches, StorageView& cached_keys,
                                       StorageView& cached_values, void* scores, dim_t first, dim_t count,
                                       dim_t row_begin) {
      const dim_t heads = cached_keys.dim(1), capacity = cached_keys.dim(2), depth = cached_keys.dim(3);
      const dim_t row_cache = heads * capacity * depth;
      float16_t* k = cached_keys.data<float16_t>();
      float16_t* v = cached_values.data<float16_t>();
      const int t = static_cast<int>(caches.time + 1);
      return {k + first * row_cache, v + first * row_cache, k, v, scores, static_cast<int32_t>(count), t,
              static_cast<int32_t>(caches.shared), static_cast<int32_t>(row_begin), static_cast<int32_t>(capacity),
              cuda::slot_scores_recipe(t), cuda::slot_output_recipe(t), cuda::slot_scores_mma(t) ? 1 : 0,
              cuda::slot_output_mma(t) ? 1 : 0};
    }
#endif

    void capacity_attention_parts(const std::vector<CapacityPart>& parts, const StorageView& queries,
                                  StorageView& keys, StorageView& values, float scale, StorageView& context) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const dim_t heads = queries.dim(1), depth = queries.dim(3);
      // Each part's step into its caches, then its scores' room in one buffer for all the parts.
      std::vector<dim_t> offsets;
      dim_t total = 0;
      for (const CapacityPart& part : parts) {
        capacity_append(*part.caches, *part.cached_keys, *part.cached_values,
                        rows_view(keys, part.row_begin, part.rows), rows_view(values, part.row_begin, part.rows));
        offsets.push_back(total);
        total += part.rows * heads * (part.caches->time + 1);
      }
      StorageView all_scores({total}, queries.dtype(), queries.device());
      float16_t* p = all_scores.data<float16_t>();
      std::vector<StorageView> scores;                       // each part's [rows, heads, 1, t], for the softmax
      std::vector<cuda::SlotAttention> fused;
      struct Own { const CapacityPart* part; dim_t first, count, t; float16_t* scores; };
      std::vector<Own> own;
      dim_t max_time = 0;
      for (size_t i = 0; i < parts.size(); ++i) {
        const CapacityPart& part = parts[i];
        const dim_t t = part.caches->time + 1;
        max_time = std::max(max_time, t);
        scores.emplace_back(queries.dtype(), queries.device());
        scores.back().view(p + offsets[i], {part.rows, heads, 1, t});
        for (const auto& [first, count] : part.groups) {
          float16_t* group_scores = p + offsets[i] + first * heads * t;
          if (cuda::slot_attention_enabled()
              && cuda::slot_attention_applies(static_cast<int>(count), static_cast<int>(heads),
                                              static_cast<int>(depth), static_cast<int>(t)))
            fused.push_back(capacity_group(*part.caches, *part.cached_keys, *part.cached_values, group_scores, first,
                                           count, part.row_begin + first));
          else
            own.push_back({&part, first, count, t, group_scores});
        }
      }
      const StorageView table = slot_table(fused);
      const auto* entries = reinterpret_cast<const cuda::SlotAttention*>(table.data<int32_t>());
      const float16_t* q = queries.data<float16_t>();
      if (context.shape() != queries.shape())                 // the step's context, the beam parts' rows in it
        throw std::logic_error("capacity_attention_parts needs the step's context");
      float16_t* out = context.data<float16_t>();
      const cuda::ClipGroupsPause alone;                     // each own group one call
      if (!fused.empty())
        cuda::slot_attention_scores(entries, static_cast<int>(fused.size()), q, static_cast<int>(heads),
                                    static_cast<int>(max_time), scale);
      for (const Own& o : own) {
        const dim_t capacity = o.part->cached_keys->dim(2), row_cache = heads * capacity * depth;
        primitives<Device::CUDA>::gemm_batch_strided(false, true, 1, o.t, depth, scale,
                                                     q + (o.part->row_begin + o.first) * heads * depth, depth, depth,
                                                     o.part->cached_keys->data<float16_t>() + o.first * row_cache,
                                                     depth, capacity * depth, 0.f, o.scores, o.t, o.t,
                                                     o.count * heads);
      }
      softmax_parts(scores);
      if (!fused.empty())
        cuda::slot_attention_output(entries, static_cast<int>(fused.size()), out, static_cast<int>(heads));
      for (const Own& o : own) {
        const dim_t capacity = o.part->cached_values->dim(2), row_cache = heads * capacity * depth;
        primitives<Device::CUDA>::gemm_batch_strided(false, false, 1, depth, o.t, 1.f, o.scores, o.t, o.t,
                                                     o.part->cached_values->data<float16_t>() + o.first * row_cache,
                                                     depth, capacity * depth, 0.f,
                                                     out + (o.part->row_begin + o.first) * heads * depth, depth,
                                                     depth, o.count * heads);
      }
#else
      (void)parts; (void)queries; (void)keys; (void)values; (void)scale; (void)context;
      throw std::logic_error("Capacity caches need CUDA");
#endif
    }

  }
}

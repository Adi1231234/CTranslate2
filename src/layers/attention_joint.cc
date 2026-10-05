#include "ctranslate2/layers/attention.h"

#include <stdexcept>

#include "ctranslate2/ops/ops.h"
#include "cross_attention_fused.h"
#include "dot_product_attention.h"
#include "joint_step.h"
#include "kv_cache.h"
#include "split_heads_fused.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/clip_groups.h"
#endif

namespace ctranslate2 {
  namespace layers {

    void MultiHeadAttention::joint_attention(const JointStep& joint, StorageView& fused_proj, bool fused_q,
                                             StorageView& context) const {
      if (!fused_q || _num_heads_kv != _num_heads || _merge_time_and_head_dims || _q_norm || _k_norm || _v_norm
          || _rotary_embeddings || _relative_attention_bias || _relative_position_keys
          || _relative_asymmetric_position_keys || _relative_position_values || _alibi || _sliding_window > 0
          || _tensor_parallel)
        throw std::logic_error("A joint decoding step supports the plain multi-head attention only");
      const Device device = fused_proj.device();
      const DataType dtype = fused_proj.dtype();

      if (!_self_attention) {
        // process_cross_attention's head split for all the clips ([clips, heads, beams, depth]), then each part's
        // clips against its memory keys and values with its own batch's arithmetic, in one launch.
        const dim_t beams = fused_proj.dim(0) / joint.clips;
        std::vector<dim_t> part_clips;
        part_clips.reserve(joint.parts.size());
        for (const auto& part : joint.parts) {
          if (part.rows != part.clips * beams)
            throw std::logic_error("A joint decoding step needs the same beams in every part");
          part_clips.push_back(part.clips);
        }
        StorageView queries_proj(dtype, device);
        split_heads_with_bias(fused_proj, _linear[0].bias(), {&queries_proj}, _num_heads, beams);
        cross_attention_joint(queries_proj, joint.memory(joint.layer), part_clips, _queries_scale, context);
        combine_heads(context, _num_heads, nullptr, beams, /*heads_combined=*/true);
        return;
      }

      // Self-attention: each part's rows on its own caches, as its own step (one batch per call, no clip groups).
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const cuda::ClipGroupsPause no_groups;
#endif
      std::vector<StorageView> contexts;
      contexts.reserve(joint.parts.size());
      for (const auto& part : joint.parts) {
        StorageView proj = rows_view(fused_proj, part.row_begin, part.rows);
        StorageView queries_proj(dtype, device);
        StorageView keys_proj(dtype, device);
        StorageView values_proj(dtype, device);
        split_heads_with_bias(proj, _linear[0].bias(), {&queries_proj, &keys_proj, &values_proj}, _num_heads);

        StorageView& cached_keys = *part.self_keys[joint.layer];
        StorageView& cached_values = *part.self_values[joint.layer];
        if (cached_keys.empty())
          throw std::logic_error("A joint decoding step needs every part's self-attention cache");
        if (part.cache_reorder) {
          reorder_and_append(cached_keys, cached_values, *part.cache_reorder, keys_proj, values_proj);
        } else {
          const ops::Concat concat_op(_cache_time_dim);
          StorageView tmp(dtype, device);
          tmp = std::move(cached_keys);
          concat_op({&tmp, &keys_proj}, cached_keys);
          tmp = std::move(cached_values);
          concat_op({&tmp, &values_proj}, cached_values);
        }

        contexts.emplace_back(dtype, device);
        StorageView& part_context = contexts.back();
        const bool heads_combined = dot_product_attention(queries_proj, cached_keys, cached_values,
                                                          /*values_lengths=*/nullptr, nullptr, nullptr, nullptr,
                                                          nullptr, 0, 0, 0, part_context, /*attention=*/nullptr,
                                                          /*return_normalized_attention=*/true, _queries_scale,
                                                          _is_decoder, /*with_cache=*/true, /*beam_size=*/1,
                                                          nullptr, nullptr);
        combine_heads(part_context, _num_heads, nullptr, 1, heads_combined);
      }
      std::vector<const StorageView*> inputs;
      inputs.reserve(contexts.size());
      for (const auto& part_context : contexts)
        inputs.push_back(&part_context);
      ops::Concat(0)(inputs, context);
    }

  }
}

#include "ctranslate2/layers/attention.h"

#include <algorithm>
#include <stdexcept>

#include "ctranslate2/ops/ops.h"
#include "cross_attention_fused.h"
#include "dot_product_attention.h"
#include "joint_step.h"
#include "joint_parts.h"
#include "slot_attention.h"
#include "slot_cache.h"
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
        // Each part's clips against its memory keys and values with its own batch's arithmetic, in one launch that
        // reads the queries from the projection with their bias (process_cross_attention's head split's values).
        const dim_t beams = fused_proj.dim(0) / joint.clips;
        std::vector<dim_t> part_clips;
        part_clips.reserve(joint.parts.size());
        for (const auto& part : joint.parts) {
          if (part.rows != part.clips * beams)
            throw std::logic_error("A joint decoding step needs the same beams in every part");
          part_clips.push_back(part.clips);
        }
        cross_attention_joint(fused_proj, _linear[0].bias(), _num_heads, joint.memory(joint.layer), part_clips,
                              _queries_scale, context);
        combine_heads(context, _num_heads, nullptr, beams, /*heads_combined=*/true);
        return;
      }

      // Self-attention: the head split of all the rows (each row's own values), every part's beam order and new
      // step into its caches in one launch (data movement), then each part's attention on its own caches as its own
      // step (one batch per call, no clip groups): dot_product_attention's three ops on a decoder step, the scores
      // MatMul and the values MatMul part by part (cuBLAS's arithmetic depends on a call's batch), the softmax of
      // every part's scores in one launch where it applies (a row's arithmetic depends on its length only); each
      // part's values product written in its own rows of the context (no join). A part in slots (slot_cache.h) has
      // its step appended to its slots, and its products run over its slots, the queries and outputs permuted.
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const cuda::ClipGroupsPause no_groups;
#endif
      StorageView all_queries(dtype, device);
      StorageView all_keys(dtype, device);
      StorageView all_values(dtype, device);
      split_heads_with_bias(fused_proj, _linear[0].bias(), {&all_queries, &all_keys, &all_values}, _num_heads);
      append_parts(joint, all_keys, all_values);             // the parts not in slots
      slot_append(joint, all_keys, all_values);
      StorageView slot_q(dtype, device), slot_out(dtype, device);
      if (joint.slot_parts > 0) {
        slot_q = StorageView(all_queries.shape(), dtype, device);
        slot_out = StorageView(all_queries.shape(), dtype, device);
        slot_queries(joint, all_queries, slot_q);
      }
      context = StorageView(all_queries.shape(), dtype, device);   // [rows, heads, 1, depth]
      std::vector<StorageView> scores;                       // allocated on the thread's own stream
      scores.reserve(joint.parts.size());
      const auto fused = [&](size_t p) { return joint.parts[p].scores_offset >= 0; };   // slot_attention.h
      for (size_t p = 0; p < joint.parts.size(); ++p) {
        const auto& part = joint.parts[p];
        const dim_t time = part.slots ? slot_time(joint, p) : part.self_keys[joint.layer]->dim(2);
        if (fused(p))
          scores.push_back(slot_scores_view(joint, p, _num_heads, time));
        else
          scores.emplace_back(Shape{part.rows, _num_heads, 1, time}, dtype, device);
      }
      const ops::MatMul keys_matmul(/*trans_a=*/false, /*trans_b=*/true, _queries_scale);
      const ops::MatMul values_matmul;
      const auto keys_product = [&](size_t p) {
        const auto& part = joint.parts[p];
        if (part.slots)
          slot_scores(joint, p, slot_q, _queries_scale, scores[p]);
        else
          keys_matmul(rows_view(all_queries, part.row_begin, part.rows), *part.self_keys[joint.layer], scores[p]);
      };
      const auto values_product = [&](size_t p) {
        const auto& part = joint.parts[p];
        StorageView part_context = rows_view(context, part.row_begin, part.rows);
        if (part.slots)
          slot_values(joint, p, scores[p], slot_out);
        else
          values_matmul(scores[p], *part.self_values[joint.layer], part_context);
      };
      for (size_t p = 0; p < joint.parts.size(); ++p)
        if (!fused(p))
          keys_product(p);
      if (joint.slot_fused > 0)
        fused_slot_scores(joint, slot_q, _queries_scale);
      softmax_parts(scores);
      for (size_t p = 0; p < joint.parts.size(); ++p)
        if (!fused(p))
          values_product(p);
      if (joint.slot_fused > 0)
        fused_slot_output(joint, slot_out);
      if (joint.slot_parts > 0)
        slot_context(joint, slot_out, context);
      combine_heads(context, _num_heads, nullptr, 1, /*heads_combined=*/false);   // one step: a reshape
    }

  }
}

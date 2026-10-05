#include "ctranslate2/layers/attention.h"

#include <algorithm>
#include <stdexcept>

#include "ctranslate2/ops/ops.h"
#include "cross_attention_fused.h"
#include "dot_product_attention.h"
#include "joint_step.h"
#include "kv_cache.h"
#include "split_heads_fused.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/cache_reorder.h"
#  include "cuda/clip_groups.h"
#  include "cuda/softmax_parts.h"
#endif

namespace ctranslate2 {
  namespace layers {

    // Each part's self-attention caches with its beam order applied and the step's keys and values (rows of keys and
    // values, [rows, heads, 1, depth]) appended: one launch for all the parts where it applies (CUDA, fp16), else
    // reorder_and_append or Concat part by part. The same values either way (data movement).
    static void append_parts(const JointStep& joint, StorageView& keys, StorageView& values) {
      const Device device = keys.device();
      const DataType dtype = keys.dtype();
      for (const auto& part : joint.parts)
        if (part.self_keys[joint.layer]->empty())
          throw std::logic_error("A joint decoding step needs every part's self-attention cache");
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const dim_t heads = keys.dim(1), depth = keys.dim(3);
      bool fused = device == Device::CUDA && dtype == DataType::FLOAT16 && keys.dim(2) == 1
        && joint.parts.size() <= static_cast<size_t>(cuda::CacheParts::max_parts)
        && cuda::cache_reorder_supported(keys.buffer(), depth, keys.item_size())
        && cuda::cache_reorder_supported(values.buffer(), depth, values.item_size());
      for (const auto& part : joint.parts) {
        const StorageView& cache = *part.self_keys[joint.layer];
        fused = fused && cache.rank() == 4 && cache.dim(1) == heads && cache.dim(3) == depth
          && cache.dim(0) == (part.cache_reorder ? cache.dim(0) : part.rows)
          && (!part.cache_reorder || (part.cache_reorder->device() == Device::CUDA
                                      && part.cache_reorder->size() == part.rows))
          && (part.row_begin * heads * depth * keys.item_size()) % 16 == 0;
      }
      if (fused) {
        cuda::CacheParts parts;
        std::vector<StorageView> out;                        // each part's new keys and values
        out.reserve(2 * joint.parts.size());
        for (const auto& part : joint.parts) {
          StorageView& cache_keys = *part.self_keys[joint.layer];
          StorageView& cache_values = *part.self_values[joint.layer];
          const dim_t time = cache_keys.dim(2);
          out.emplace_back(Shape{part.rows, heads, time + 1, depth}, dtype, device);
          out.emplace_back(Shape{part.rows, heads, time + 1, depth}, dtype, device);
          const int p = parts.count++;
          parts.cache[p][0] = cache_keys.buffer();
          parts.cache[p][1] = cache_values.buffer();
          parts.fresh[p][0] = keys.data<float16_t>() + part.row_begin * heads * depth;
          parts.fresh[p][1] = values.data<float16_t>() + part.row_begin * heads * depth;
          parts.out[p][0] = out[2 * p].buffer();
          parts.out[p][1] = out[2 * p + 1].buffer();
          parts.order[p] = part.cache_reorder ? part.cache_reorder->data<int32_t>() : nullptr;
          parts.rows[p] = static_cast<int>(part.rows);
          parts.time[p] = static_cast<int>(time);
        }
        cuda::reorder_append_parts(parts, heads, depth);
        for (size_t p = 0; p < joint.parts.size(); ++p) {
          *joint.parts[p].self_keys[joint.layer] = std::move(out[2 * p]);
          *joint.parts[p].self_values[joint.layer] = std::move(out[2 * p + 1]);
        }
        return;
      }
#endif
      for (const auto& part : joint.parts) {
        StorageView& cached_keys = *part.self_keys[joint.layer];
        StorageView& cached_values = *part.self_values[joint.layer];
        StorageView part_keys = rows_view(keys, part.row_begin, part.rows);
        StorageView part_values = rows_view(values, part.row_begin, part.rows);
        if (part.cache_reorder) {
          reorder_and_append(cached_keys, cached_values, *part.cache_reorder, part_keys, part_values);
        } else {
          StorageView tmp(dtype, device);
          tmp = std::move(cached_keys);
          ops::Concat(2)({&tmp, &part_keys}, cached_keys);
          tmp = std::move(cached_values);
          ops::Concat(2)({&tmp, &part_values}, cached_values);
        }
      }
    }

    // ops::SoftMax of each part's scores, in place: one launch for all the parts where it applies (CUDA, fp16, the
    // warp kernel's lengths), else part by part. The same values either way (cuda/softmax_parts.h).
    static void softmax_parts(std::vector<StorageView>& scores) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      bool fused = !scores.empty() && scores.size() <= static_cast<size_t>(cuda::SoftmaxParts::max_parts);
      cuda::SoftmaxParts parts;
      unsigned rows = 0;
      for (StorageView& s : scores) {
        fused = fused && s.device() == Device::CUDA && s.dtype() == DataType::FLOAT16
          && cuda::softmax_parts_supported(s.dim(-1));
        if (!fused)
          break;
        rows += static_cast<unsigned>(s.size() / s.dim(-1));
        parts.data[parts.count] = s.buffer();
        parts.rows_end[parts.count] = rows;
        parts.cols[parts.count++] = static_cast<unsigned>(s.dim(-1));
      }
      if (fused) {
        cuda::softmax_parts(parts);
        return;
      }
#endif
      for (StorageView& s : scores)
        ops::SoftMax()(s, nullptr, s);
    }

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
      // part's values product written in its own rows of the context (no join).
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const cuda::ClipGroupsPause no_groups;
#endif
      StorageView all_queries(dtype, device);
      StorageView all_keys(dtype, device);
      StorageView all_values(dtype, device);
      split_heads_with_bias(fused_proj, _linear[0].bias(), {&all_queries, &all_keys, &all_values}, _num_heads);
      append_parts(joint, all_keys, all_values);

      std::vector<StorageView> scores;
      scores.reserve(joint.parts.size());
      const ops::MatMul keys_matmul(/*trans_a=*/false, /*trans_b=*/true, _queries_scale);
      for (const auto& part : joint.parts) {
        scores.emplace_back(dtype, device);
        keys_matmul(rows_view(all_queries, part.row_begin, part.rows), *part.self_keys[joint.layer], scores.back());
      }
      softmax_parts(scores);
      context = StorageView(all_queries.shape(), dtype, device);   // [rows, heads, 1, depth]
      const ops::MatMul values_matmul;
      for (size_t p = 0; p < joint.parts.size(); ++p) {
        StorageView part_context = rows_view(context, joint.parts[p].row_begin, joint.parts[p].rows);
        values_matmul(scores[p], *joint.parts[p].self_values[joint.layer], part_context);
      }
      combine_heads(context, _num_heads, nullptr, 1, /*heads_combined=*/false);   // one step: a reshape
    }

  }
}

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
#  include "cuda/copy_parts.h"
#  include "cuda/utils.h"
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

    // The parts' outputs one after the other along the rows: one launch where it applies (CUDA), where ops::Concat
    // copies each part on its own.
    static void join_rows(std::vector<StorageView>& parts, StorageView& out) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      bool fused = !parts.empty() && parts.size() <= static_cast<size_t>(cuda::CopyParts::max_parts);
      cuda::CopyParts copy;
      dim_t rows = 0;
      for (const StorageView& part : parts) {
        fused = fused && part.device() == Device::CUDA && part.dtype() == parts[0].dtype()
          && part.rank() == parts[0].rank() && part.size() / part.dim(0) == parts[0].size() / parts[0].dim(0)
          && cuda::copy_parts_supported(part.buffer(), part.size() * part.item_size());
        if (!fused)
          break;
        copy.src[copy.count] = part.buffer();
        copy.bytes[copy.count++] = part.size() * part.item_size();
        rows += part.dim(0);
      }
      if (fused) {
        Shape shape = parts[0].shape();
        shape[0] = rows;
        out.resize(std::move(shape));
        if (cuda::copy_parts_supported(out.buffer(), out.size() * out.item_size())) {
          cuda::copy_parts(copy, out.buffer());
          return;
        }
      }
#endif
      std::vector<const StorageView*> inputs;
      inputs.reserve(parts.size());
      for (const auto& part : parts)
        inputs.push_back(&part);
      ops::Concat(0)(inputs, out);
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

      // Self-attention: the head split of all the rows (each row's own values), every part's beam order and new
      // step into its caches in one launch (data movement), then each part's attention on its own caches as its own
      // step (one batch per call, no clip groups).
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const cuda::ClipGroupsPause no_groups;
#endif
      StorageView all_queries(dtype, device);
      StorageView all_keys(dtype, device);
      StorageView all_values(dtype, device);
      split_heads_with_bias(fused_proj, _linear[0].bias(), {&all_queries, &all_keys, &all_values}, _num_heads);
      append_parts(joint, all_keys, all_values);

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      // The parts' attentions are independent: on side streams (CT2_SIDE_STREAMS), after the cache update, and
      // joined back before their contexts are (cuda/utils.h).
      const int sides = std::min<int>(cuda::side_streams(), static_cast<int>(joint.parts.size()));
      static thread_local std::vector<cudaEvent_t> events;
      while (static_cast<int>(events.size()) < sides + 1) {
        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        events.push_back(event);
      }
      cudaStream_t main_stream = cuda::get_cuda_stream();
      if (sides > 1)
        CUDA_CHECK(cudaEventRecord(events[sides], main_stream));
#endif
      std::vector<StorageView> contexts;
      contexts.reserve(joint.parts.size());
      for (size_t p = 0; p < joint.parts.size(); ++p) {
        const auto& part = joint.parts[p];
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
        std::unique_ptr<cuda::UseSideStreamInScope> side;
        if (sides > 1) {
          side = std::make_unique<cuda::UseSideStreamInScope>(static_cast<int>(p % sides));
          if (p < static_cast<size_t>(sides))
            CUDA_CHECK(cudaStreamWaitEvent(cuda::get_cuda_stream(), events[sides], 0));
        }
#endif
        StorageView queries_proj = rows_view(all_queries, part.row_begin, part.rows);
        StorageView& cached_keys = *part.self_keys[joint.layer];
        StorageView& cached_values = *part.self_values[joint.layer];

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
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      for (int s = 0; sides > 1 && s < sides; ++s) {
        {
          const cuda::UseSideStreamInScope side(s);
          CUDA_CHECK(cudaEventRecord(events[s], cuda::get_cuda_stream()));
        }
        CUDA_CHECK(cudaStreamWaitEvent(main_stream, events[s], 0));
      }
#endif
      join_rows(contexts, context);
    }

  }
}

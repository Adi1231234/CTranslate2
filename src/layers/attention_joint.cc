#include "ctranslate2/layers/attention.h"

#include <algorithm>
#include <stdexcept>

#include "ctranslate2/ops/ops.h"
#include "cross_attention_fused.h"
#include "dot_product_attention.h"
#include "joint_step.h"
#include "joint_parts.h"
#include "split_heads_fused.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/clip_groups.h"
#  include "cuda/utils.h"
#  include "env.h"
#endif

namespace ctranslate2 {
  namespace layers {

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
    // CT2_JOINT_STREAMS=<n> (default 0: off): the parts' self-attention products on n side streams of the thread
    // (cuda::SideStreamScope, a cuBLAS handle each), the parts dealt to them in turn. Each part runs the very calls,
    // kernels and data it runs on the thread's own stream; only side by side: with a part a window (long
    // recordings), a step's ~30 parts' calls of 100 entries each fill a small part of the GPU one after the other
    // (long5's profile: ~2/3 of the GPU's time in the per-part self-attention).
    static int joint_streams() {
      static const int streams = read_int_from_env("CT2_JOINT_STREAMS", 0);
      return streams;
    }

    // run(p) for p in [0, count) on the side streams, after the thread's stream's work so far and before its work
    // from now on (events).
    template <typename Run>
    static void side_by_side(size_t count, Run&& run) {
      const int streams = static_cast<int>(std::min<size_t>(joint_streams(), count));
      static thread_local std::vector<cudaEvent_t> events;   // [0]: the fork; [s]: side stream s done
      while (events.size() <= static_cast<size_t>(streams)) {
        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        events.push_back(event);
      }
      const cudaStream_t own = cuda::get_cuda_stream();
      CUDA_CHECK(cudaEventRecord(events[0], own));
      for (int s = 1; s <= streams; ++s) {
        const cuda::SideStreamScope side(s);
        CUDA_CHECK(cudaStreamWaitEvent(cuda::get_cuda_stream(), events[0], 0));
        for (size_t p = s - 1; p < count; p += streams)
          run(p);
        CUDA_CHECK(cudaEventRecord(events[s], cuda::get_cuda_stream()));
      }
      for (int s = 1; s <= streams; ++s)
        CUDA_CHECK(cudaStreamWaitEvent(own, events[s], 0));
    }
#endif

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
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      // CT2_JOINT_STREAMS=<n>: the same three ops of each part on n side streams (side_by_side).
      if (device == Device::CUDA && joint_streams() > 0 && joint.parts.size() > 1) {
        context = StorageView(all_queries.shape(), dtype, device);
        for (const auto& part : joint.parts)                 // allocated on the thread's own stream, before the fork
          scores.emplace_back(Shape{part.rows, _num_heads, 1, part.self_keys[joint.layer]->dim(2)}, dtype, device);
        const ops::MatMul values_matmul;
        side_by_side(joint.parts.size(), [&](size_t p) {
          const auto& part = joint.parts[p];
          keys_matmul(rows_view(all_queries, part.row_begin, part.rows), *part.self_keys[joint.layer], scores[p]);
          ops::SoftMax()(scores[p], nullptr, scores[p]);
          StorageView part_context = rows_view(context, part.row_begin, part.rows);
          values_matmul(scores[p], *part.self_values[joint.layer], part_context);
        });
        combine_heads(context, _num_heads, nullptr, 1, /*heads_combined=*/false);
        return;
      }
#endif
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

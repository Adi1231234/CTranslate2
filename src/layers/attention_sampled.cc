#include "attention_sampled.h"

#include <stdexcept>

#include "capacity_cache.h"
#include "dot_product_attention.h"
#include "split_heads_fused.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/clip_groups.h"
#  include "cuda/shared_memory_rows.h"
#endif

namespace ctranslate2 {
  namespace layers {

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
    // The groups the part's search's decoder call has (cuda::make_clip_groups of its rows).
    static cuda::ClipGroups part_groups(const SampledRows& rows) {
      cuda::ClipGroups groups;
      groups.clips = rows.group_rows;
      groups.total = rows.rows;
      return groups;
    }
#endif

    void sampled_self_attention(const JointStep& joint, const JointStep::Part& part, StorageView& queries,
                                StorageView& keys, StorageView& values, float scale, StorageView& context) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const SampledRows& rows = *part.sampled;
      const cuda::ClipGroupsScope groups(part_groups(rows));
      const CapacityCacheScope capacity(rows.capacity);
      StorageView part_queries = rows_view(queries, part.row_begin, part.rows);
      StorageView part_keys = rows_view(keys, part.row_begin, part.rows);
      StorageView part_values = rows_view(values, part.row_begin, part.rows);
      StorageView part_context = rows_view(context, part.row_begin, part.rows);
      capacity_attention(part_queries, part_keys, part_values, scale, *part.self_keys[joint.layer],
                         *part.self_values[joint.layer], part_context);
      if (part_context.buffer() != rows_view(context, part.row_begin, part.rows).buffer())
        throw std::logic_error("A joint step's greedy part's self-attention left its rows");
#else
      (void)joint; (void)part; (void)queries; (void)keys; (void)values; (void)scale; (void)context;
      throw std::logic_error("A joint step's greedy part requires CUDA");
#endif
    }

    void sampled_cross_attention(const JointStep& joint, const JointStep::Part& part, StorageView& proj,
                                 const StorageView* bias, dim_t heads, float scale, StorageView& context) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      const SampledRows& rows = *part.sampled;
      const cuda::ClipGroupsScope groups(part_groups(rows));
      const cuda::SharedMemoryRows shared{rows.row_input, rows.rows, rows.inputs};
      const cuda::SharedMemoryRowsScope shared_scope(shared);
      StorageView part_proj = rows_view(proj, part.row_begin, part.rows);
      StorageView queries(proj.dtype(), proj.device());
      split_heads_with_bias(part_proj, bias, {&queries}, heads, /*beam_size=*/1);
      // Its own output, then copied into its rows: with one row left of its input (no shared rows then), the
      // stock path's first MatMul writes the scores into the output (long24: a view of the rows would move).
      StorageView part_out(proj.dtype(), proj.device());
      const bool heads_combined = dot_product_attention(queries, *part.memory_keys[joint.layer],
                                                        *part.memory_values[joint.layer], nullptr, nullptr, nullptr,
                                                        nullptr, nullptr, 0, 0, 0, part_out, nullptr,
                                                        /*return_normalized_attention=*/true, scale,
                                                        /*is_decoder=*/true, /*with_cache=*/true, /*beam_size=*/1,
                                                        nullptr, nullptr);
      (void)heads_combined;                                  // one query a row: either layout is [rows, heads x depth]
      StorageView part_context = rows_view(context, part.row_begin, part.rows);
      if (part_out.size() != part_context.size())
        throw std::logic_error("A joint step's greedy part's cross-attention has another shape");
      part_context.copy_from(part_out);
      if (part_context.buffer() != rows_view(context, part.row_begin, part.rows).buffer())
        throw std::logic_error("A joint step's greedy part's cross-attention left its rows");
#else
      (void)joint; (void)part; (void)proj; (void)bias; (void)heads; (void)scale; (void)context;
      throw std::logic_error("A joint step's greedy part requires CUDA");
#endif
    }

  }
}

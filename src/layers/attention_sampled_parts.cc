#include "attention_sampled.h"

#include "ctranslate2/ops/ops.h"
#include "capacity_cache.h"
#include "env.h"
#include "split_heads_fused.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/ladder_cross.h"
#endif

namespace ctranslate2 {
  namespace layers {

    // A greedy part's groups as its own call has them (for_each_clip_group of its rows under its clip groups: each
    // group still decoding, in order), or none where its rows are not all in groups.
    static std::vector<std::pair<dim_t, dim_t>> part_groups(const JointStep::Part& part) {
      std::vector<std::pair<dim_t, dim_t>> groups;
      dim_t first = 0;
      for (const dim_t count : part.sampled->group_rows) {
        if (count == 0)
          continue;
        groups.emplace_back(first, count);
        first += count;
      }
      if (first != part.rows)
        groups.clear();
      return groups;
    }

    // CT2_SAMPLED_PARTS_SELF / _CROSS=0: those parts one by one, as their searches alone (A/B, never a result).
    static bool parts_at_once(const char* name) {
      return read_bool_from_env(name, true);
    }

    void sampled_self_attention_all(const JointStep& joint, StorageView& queries, StorageView& keys,
                                    StorageView& values, float scale, StorageView& context) {
      static const bool at_once = parts_at_once("CT2_SAMPLED_PARTS_SELF");
      std::vector<CapacityPart> parts;
      for (const auto& part : joint.parts) {
        if (!part.sampled)
          continue;
        auto groups = at_once && part.sampled->capacity && part.sampled->rows == part.rows
          ? part_groups(part) : decltype(part_groups(part))();
        if (groups.empty()) {                                // as its search alone
          sampled_self_attention(joint, part, queries, keys, values, scale, context);
          continue;
        }
        parts.push_back({part.sampled->capacity, part.self_keys[joint.layer], part.self_values[joint.layer],
                         part.row_begin, part.rows, std::move(groups)});
      }
      if (!parts.empty())
        capacity_attention_parts(parts, queries, keys, values, scale, context);
    }

    void sampled_cross_attention_all(const JointStep& joint, StorageView& proj, const StorageView* bias, dim_t heads,
                                     float scale, StorageView& context) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      // The parts whose rows read one clip's memory through shared rows (more rows than inputs: a part of one row
      // and one input runs the fused attention, attention.cc) with recovered arithmetic (cuda/ladder_cross.h).
      std::vector<cuda::LadderMemory> ladders;
      std::vector<const JointStep::Part*> rest;
      dim_t first_row = -1, end_row = 0;
      for (const auto& part : joint.parts) {
        if (!part.sampled)
          continue;
        if (first_row < 0)
          first_row = part.row_begin;
        end_row = part.row_begin + part.rows;
        const StorageView& k = *part.memory_keys[joint.layer];
        const StorageView& v = *part.memory_values[joint.layer];
        const auto groups = part_groups(part);
        cuda::LadderMemory ladder{nullptr, nullptr, 0, {}};
        for (const auto& group : groups)
          ladder.groups.push_back(group.second);
        static const bool at_once = parts_at_once("CT2_SAMPLED_PARTS_CROSS");
        const bool applies = at_once && part.sampled->inputs == 1 && part.rows > 1 && !groups.empty()
          && k.device() == Device::CUDA && k.dtype() == DataType::FLOAT16 && k.rank() == 4 && k.dim(0) == 1
          && k.dim(1) == heads && k.dim(2) == 1500 && k.dim(3) == 64 && v.shape() == k.shape()
          && v.dtype() == k.dtype() && cuda::ladders_supported({ladder});
        if (!applies) {
          rest.push_back(&part);
          continue;
        }
        ladder.keys = reinterpret_cast<const __half*>(k.data<float16_t>());
        ladder.values = reinterpret_cast<const __half*>(v.data<float16_t>());
        ladder.row_begin = part.row_begin;                   // from first_row below
        ladders.push_back(std::move(ladder));
      }
      if (!ladders.empty()) {
        for (auto& ladder : ladders)
          ladder.row_begin -= first_row;
        const dim_t rows = end_row - first_row;
        StorageView queries(proj.dtype(), proj.device());
        split_heads_with_bias(rows_view(proj, first_row, rows), bias, {&queries}, heads, /*beam_size=*/1);
        StorageView attn({rows, heads, 1, 1500}, proj.dtype(), proj.device());
        __half* scores = reinterpret_cast<__half*>(attn.data<float16_t>());
        cuda::ladders_cross_scores(ladders, reinterpret_cast<const __half*>(queries.data<float16_t>()), scores, heads,
                                   scale);
        ops::SoftMax()(attn, nullptr, attn);                 // a row's softmax depends on that row only
        StorageView out = rows_view(context, first_row, rows);
        cuda::ladders_cross_output(ladders, scores, reinterpret_cast<__half*>(out.data<float16_t>()), heads);
      }
      for (const JointStep::Part* part : rest)
        sampled_cross_attention(joint, *part, proj, bias, heads, scale, context);
#else
      for (const auto& part : joint.parts)
        if (part.sampled)
          sampled_cross_attention(joint, part, proj, bias, heads, scale, context);
#endif
    }

  }
}

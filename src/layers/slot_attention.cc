#include "slot_attention.h"

#include <cstring>

#include "slot_cache.h"
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include "cuda/slot_attention.h"
#endif

namespace ctranslate2 {
  namespace layers {

    void prepare_slot_attention(JointStep& joint, const std::vector<JointStep::Part*>& slot_parts) {
      joint.slot_fused = false;
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      if (slot_parts.empty() || !cuda::slot_attention_enabled())
        return;
      const SlotCache& first = *slot_parts[0]->slots;
      const dim_t heads = first.keys[0].dim(1), depth = first.keys[0].dim(3);
      dim_t total = 0;
      for (JointStep::Part* part : slot_parts) {
        const SlotCache& s = *part->slots;
        const dim_t time = s.time + 1;                       // this step's position too
        if (!cuda::slot_attention_applies(s.rows, heads, depth, time))
          return;                                            // the parts' own calls
        part->scores_offset = total;
        total += s.rows * heads * time;
      }
      joint.slot_scores = StorageView({total}, DataType::FLOAT16, Device::CUDA);
      auto* scores = static_cast<char*>(joint.slot_scores.buffer());
      std::vector<cuda::SlotAttention> table;
      const size_t layers = first.keys.size();
      table.reserve(layers * slot_parts.size());
      for (size_t l = 0; l < layers; ++l)
        for (JointStep::Part* part : slot_parts) {
          const SlotCache& s = *part->slots;
          const int time = static_cast<int>(s.time + 1);
          table.push_back({s.keys[l].buffer(), s.values[l].buffer(),
                           scores + part->scores_offset * joint.slot_scores.item_size(),
                           static_cast<int32_t>(s.rows), time, static_cast<int32_t>(s.shared),
                           static_cast<int32_t>(part->row_begin), cuda::slot_scores_recipe(time),
                           cuda::slot_output_recipe(time), cuda::slot_scores_mma(time) ? 1 : 0,
                           cuda::slot_output_mma(time) ? 1 : 0});
        }
      std::vector<int32_t> words(table.size() * sizeof (cuda::SlotAttention) / sizeof (int32_t));
      std::memcpy(words.data(), table.data(), table.size() * sizeof (cuda::SlotAttention));
      joint.slot_attention = StorageView({static_cast<dim_t>(words.size())}, words).to(Device::CUDA);
      joint.slot_fused = true;
#endif
    }

    StorageView slot_scores_view(const JointStep& joint, size_t part, dim_t heads, dim_t time) {
      const JointStep::Part& p = joint.parts[part];
      StorageView view(DataType::FLOAT16, Device::CUDA);
      auto* base = static_cast<char*>(const_cast<StorageView&>(joint.slot_scores).buffer());
      view.view(base + p.scores_offset * joint.slot_scores.item_size(), Shape{p.rows, heads, 1, time});
      return view;
    }

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
    static const cuda::SlotAttention* layer_table(const JointStep& joint) {
      return reinterpret_cast<const cuda::SlotAttention*>(joint.slot_attention.data<int32_t>())
        + joint.layer * joint.slot_parts;
    }
#endif

    void fused_slot_scores(const JointStep& joint, const StorageView& slot_queries, float scale) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      cuda::slot_attention_scores(layer_table(joint), joint.slot_parts, slot_queries.buffer(), slot_queries.dim(1),
                                  joint.slot_max_time + 1, scale);
#else
      (void)joint; (void)slot_queries; (void)scale;
#endif
    }

    void fused_slot_output(const JointStep& joint, StorageView& slot_output) {
#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
      cuda::slot_attention_output(layer_table(joint), joint.slot_parts, slot_output.buffer(), slot_output.dim(1));
#else
      (void)joint; (void)slot_output;
#endif
    }

  }
}

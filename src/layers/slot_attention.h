#pragma once

#include <vector>

#include "ctranslate2/storage_view.h"
#include "joint_step.h"

namespace ctranslate2 {
  namespace layers {

    // The slot parts' self-attention products of a joint step in one launch per product and layer
    // (cuda/slot_attention.h), where every slot part has the arithmetic recovered for its length (CT2_SLOT_ATTENTION).
    // After prepare_slots made the parts' plans: the descriptor table and the scores buffer for all the layers.
    void prepare_slot_attention(JointStep& joint, const std::vector<JointStep::Part*>& slot_parts);
    // A slot part's scores [rows, heads, 1, time], a view of joint.slot_scores (fused steps only).
    StorageView slot_scores_view(const JointStep& joint, size_t part, dim_t heads, dim_t time);
    // Every slot part's scores (queries in slot order) and output (into slot order), this layer.
    void fused_slot_scores(const JointStep& joint, const StorageView& slot_queries, float scale);
    void fused_slot_output(const JointStep& joint, StorageView& slot_output);

  }
}

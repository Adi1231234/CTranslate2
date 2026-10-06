#pragma once

#include <vector>

#include "ctranslate2/storage_view.h"
#include "joint_step.h"

namespace ctranslate2 {
  namespace layers {

    // Each part's self-attention caches with its beam order applied and the step's keys and values (rows of keys and
    // values, [rows, heads, 1, depth]) appended: one launch for every max_parts parts where it applies (CUDA, fp16),
    // else reorder_and_append or Concat part by part. The same values either way (data movement).
    void append_parts(const JointStep& joint, StorageView& keys, StorageView& values);

    // ops::SoftMax of each part's scores, in place: one launch for every max_parts parts where it applies (CUDA,
    // fp16, the warp kernel's lengths), else part by part. The same values either way (cuda/softmax_parts.h: a
    // row's result depends on its length only).
    void softmax_parts(std::vector<StorageView>& scores);

  }
}

#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // The softmax of several fp16 score tensors in one launch (the parts' self-attentions of a joint decoding
    // step, layers/joint_step.h), in place, each row with the arithmetic ops::SoftMax gives a call with its part
    // alone (the warp kernel of ops/softmax_kernels.cuh: a row's result depends on its length only): part p's
    // rows [rows_end[p - 1], rows_end[p]) of cols[p] values at data[p].
    struct SoftmaxParts {
      static constexpr int max_parts = 16;
      int count = 0;
      void* data[max_parts];
      unsigned rows_end[max_parts];
      unsigned cols[max_parts];
    };

    // Whether ops::SoftMax runs the warp kernel for rows of `cols` fp16 values (no lengths), as softmax_parts does.
    bool softmax_parts_supported(dim_t cols);
    void softmax_parts(const SoftmaxParts& parts);

  }
}

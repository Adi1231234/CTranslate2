#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // Several device buffers copied one after the other into out in one launch (the outputs of several searches'
    // steps joined, layers/joint_step.h), where ops::Concat issues a copy per input: part p's bytes[p] bytes from
    // src[p]. Byte counts and addresses multiples of 16; at most max_parts parts.
    struct CopyParts {
      static constexpr int max_parts = 16;
      int count = 0;
      const void* src[max_parts];
      size_t bytes[max_parts];
    };
    bool copy_parts_supported(const void* p, size_t bytes);
    void copy_parts(const CopyParts& parts, void* out);

  }
}

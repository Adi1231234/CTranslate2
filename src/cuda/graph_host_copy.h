#pragma once

#include <cstddef>

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // A host to device copy of up to 64 KB while `stream` is captured (a CUDA graph step): made a kernel carrying
    // the bytes as its parameter. A copy node would read `src` when the graph runs, after the host may have freed
    // or reused it. False when not capturing: the caller copies as usual.
    // A header of its own: primitives.cu includes it and takes 45 s to compile.
    bool copy_host_bytes_if_capturing(void* dst, const void* src, size_t bytes, cudaStream_t stream);

  }
}

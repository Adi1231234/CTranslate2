#pragma once

#include <cuda_runtime.h>

namespace ctranslate2 {
  namespace cuda {

    // Where a worker thread's streams run, through green contexts (CUDA 12.4+ driver; only where kernels run
    // changes, never a result):
    //   CT2_SM_PARTITION=<decoder SMs>:<encoder SMs> (e.g. 16:20 of 36): disjoint SM sets from one split, the
    //     thread's normal stream (decoding) on the first, its low-priority stream (Whisper's encoder, see
    //     UseLowPriorityStreamInScope) on the second. One side is a multiple of 8 (whole SM groups), the other
    //     the SMs left.
    //   CT2_ENCODER_SMS=<n>: only the low-priority stream confined to n SMs; the decoder keeps the whole GPU.
    // Returns a stream for the given side and priority, or null to make a plain stream.
    cudaStream_t create_partition_stream(bool low, int priority);

  }
}

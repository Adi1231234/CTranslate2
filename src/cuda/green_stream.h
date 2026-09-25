#pragma once

#include <cuda_runtime.h>

namespace ctranslate2 {
  namespace cuda {

    // CT2_ENCODER_SMS=<n> (0, the default: off): a thread's low-priority stream (the encoder's, see
    // UseLowPriorityStreamInScope) runs its kernels on n SMs only, through a green context (CUDA 12.4+ driver),
    // while the decoder's streams keep the whole GPU. n is a multiple of 8 (whole SM groups) or the SMs left
    // after one such split (e.g. 28 or 20 of 36). Only where the kernels run changes, never a result.
    int encoder_sm_count();

    // A stream on a green context of `sms` SMs of the current device, with the given priority, or null when the
    // driver cannot make one.
    cudaStream_t create_green_stream(int sms, int priority);

  }
}

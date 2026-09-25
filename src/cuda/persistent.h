#pragma once

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // Persistent kernels: a grid of a few blocks per SM that take their work items (output tiles) from a counter
    // until none is left. A long kernel then holds a fixed share of every SM for its whole run, and the rest of
    // each SM stays free for another stream's kernels (the Whisper encoder beside the decoder, whose blocks
    // would otherwise wait for encoder blocks to finish). Which block computes an item never changes the item.

    int sm_count();

    // The work counter of the calling thread's kernels on `stream` (device memory, zero between kernels: the
    // block taking the last index resets it, see persistent.cuh). Kept for the thread's lifetime.
    unsigned* work_counter(cudaStream_t stream);

    // Blocks per SM from the environment variable `name`: 0 (the default) keeps a block per work item.
    int persistent_blocks_per_sm(const char* name);

  }
}

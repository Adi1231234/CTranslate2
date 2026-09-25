#pragma once

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // Stream-ordered allocations inside a captured step become graph memory nodes, and a captured free may
    // only release graph memory (cudaFreeAsync fails with "invalid argument" on other memory). The first
    // step's caches and the encoder output are allocated outside any graph, so their frees during a capture
    // wait until the graph is launched and are then enqueued on the same stream: freed after the same work.

    // Set by StepGraph for the thread whose stream is being captured.
    void set_step_capturing(bool capturing);
    // Called by the allocator after each allocation.
    void note_allocation(void* ptr);
    // True when the free of `ptr` is left to release_deferred().
    bool defer_free(void* ptr);
    // Enqueues the deferred frees on `stream`.
    void release_deferred(cudaStream_t stream);

  }
}

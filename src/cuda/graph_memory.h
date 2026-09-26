#pragma once

#include <cstddef>

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // A captured decoding step (cuda/graph.h) allocates from its thread's step arena instead of the pool, so the
    // graph has no memory nodes and the previous step's executable graph can be updated in place. Two arenas serve
    // alternate steps: a step's outputs (the grown attention caches, the logits) are read and freed by the next
    // step, so an arena is empty again two steps later. Each arena is one pool allocation made before the capture
    // and bump-allocated during it; freeing arena memory only counts down its live allocations (the stream orders
    // the reuse). A step whose arena still holds live data, or an allocation that does not fit, falls back to the
    // pool (stream-ordered allocations inside a capture become graph memory nodes). Frees of other memory inside a
    // capture wait until the graph is launched: cudaFreeAsync may only release graph memory there.

    // Before capturing step `step` on `stream`: false when that step's arena still holds live data.
    bool begin_step_arena(long long step, cudaStream_t stream);
    // After the capture ended: true when an allocation did not fit (the graph then has memory nodes).
    bool end_step_arena();
    // During a capture: memory from the step's arena, or null (not capturing, or full).
    void* arena_allocate(size_t size);
    // True when `ptr` is arena memory (its release is only counted).
    bool arena_free(void* ptr);
    // Allocations of the pool made inside a capture (graph memory), and frees of other memory deferred.
    void note_allocation(void* ptr);
    bool defer_free(void* ptr);
    void release_deferred(cudaStream_t stream);
    // After a decoding loop: the thread's arenas are released once empty (now, or at their last free).
    void release_arenas();

  }
}

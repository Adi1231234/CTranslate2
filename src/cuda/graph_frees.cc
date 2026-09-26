#include "cuda/graph_memory.h"

#include <mutex>
#include <unordered_set>
#include <vector>

#include "cuda/graph.h"

namespace ctranslate2 {
  namespace cuda {

    // Pool memory allocated inside a captured segment (graph memory: its free may happen inside a later capture),
    // and the frees of other memory requested during a capture, enqueued once the segment is launched.
    static std::mutex mutex;                                    // graph memory can be freed by any thread
    static std::unordered_set<void*> graph_memory;
    static thread_local std::vector<void*> deferred_frees;

    void note_allocation(void* ptr) {
      if (!capturing_step())
        return;
      const std::lock_guard<std::mutex> lock(mutex);
      graph_memory.insert(ptr);
    }

    bool defer_free(void* ptr) {
      if (!graphs_enabled())
        return false;
      bool from_graph = false;
      {
        const std::lock_guard<std::mutex> lock(mutex);
        from_graph = graph_memory.erase(ptr) > 0;
      }
      if (from_graph || !capturing_step())
        return false;
      deferred_frees.push_back(ptr);
      return true;
    }

    void release_deferred(cudaStream_t stream) {
      std::vector<void*> frees;
      frees.swap(deferred_frees);
      for (void* ptr : frees)
        CUDA_CHECK(cudaFreeAsync(ptr, stream));
    }

  }
}

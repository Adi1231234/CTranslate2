#include "cuda/graph_memory.h"

#include <mutex>
#include <unordered_set>
#include <vector>

#include "cuda/graph.h"

namespace ctranslate2 {
  namespace cuda {

    static thread_local bool step_capturing = false;
    static thread_local std::vector<void*> deferred_frees;

    // Graph memory can be freed by any thread, so the record of it is shared.
    static std::mutex graph_memory_mutex;
    static std::unordered_set<void*> graph_memory;

    void set_step_capturing(bool capturing) {
      step_capturing = capturing;
    }

    void note_allocation(void* ptr) {
      if (!step_capturing)
        return;
      const std::lock_guard<std::mutex> lock(graph_memory_mutex);
      graph_memory.insert(ptr);
    }

    bool defer_free(void* ptr) {
      if (!graphs_enabled())
        return false;
      bool from_graph = false;
      {
        const std::lock_guard<std::mutex> lock(graph_memory_mutex);
        from_graph = graph_memory.erase(ptr) > 0;
      }
      if (from_graph || !step_capturing)
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

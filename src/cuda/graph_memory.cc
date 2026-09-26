#include "cuda/graph_memory.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "cuda/graph.h"

namespace ctranslate2 {
  namespace cuda {

    struct Arena {
      char* base = nullptr;
      size_t capacity = 0, offset = 0, peak = 0;
      long long live = 0;                     // allocations not freed yet
      bool released = false;                  // the decoding loop ended: free the buffer once empty
      bool overflowed = false;
      cudaStream_t stream = nullptr;          // the owner's stream, which orders the buffer's own release
    };

    // Arena memory and graph memory can be freed by any thread, so their records are shared.
    static std::mutex mutex;
    static std::vector<std::unique_ptr<Arena>> arenas;           // never shrinks: pointers stay valid
    static std::unordered_set<void*> graph_memory;
    static thread_local Arena* own[2] = {nullptr, nullptr};
    static thread_local Arena* capturing = nullptr;
    static thread_local bool step_capturing = false;
    static thread_local std::vector<void*> deferred_frees;

    static Arena* own_arena(int i) {                              // under the lock
      if (!own[i]) {
        arenas.push_back(std::make_unique<Arena>());
        own[i] = arenas.back().get();
      }
      return own[i];
    }

    static void free_buffer(Arena& a) {                           // under the lock, a.live == 0
      if (a.base)
        CUDA_CHECK(cudaFreeAsync(a.base, a.stream));
      a.base = nullptr;
      a.capacity = 0;
    }

    bool begin_step_arena(long long step, cudaStream_t stream) {
      const std::lock_guard<std::mutex> lock(mutex);
      Arena& a = *own_arena(step % 2);
      Arena& other = *own_arena(1 - step % 2);
      if (a.live > 0)
        return false;
      a.released = other.released = false;
      const size_t want = std::max(a.peak, other.peak) + (size_t(32) << 20);   // a step grows the caches a little
      if (a.capacity < want || a.overflowed) {
        free_buffer(a);
        a.stream = stream;
        a.capacity = want + want / 2;
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&a.base), a.capacity, stream));
        a.overflowed = false;
      }
      a.offset = 0;
      a.peak = 0;
      a.stream = stream;
      capturing = &a;
      step_capturing = true;
      return true;
    }

    bool end_step_arena() {
      const bool overflowed = capturing && capturing->overflowed;
      capturing = nullptr;
      step_capturing = false;
      return overflowed;
    }

    void* arena_allocate(size_t size) {
      Arena* a = capturing;
      if (!a)
        return nullptr;
      const std::lock_guard<std::mutex> lock(mutex);
      const size_t start = (a->offset + 255) / 256 * 256;
      a->peak = std::max(a->peak, start + size);
      if (start + size > a->capacity) {
        a->overflowed = true;                                     // the pool serves it; the next arena is larger
        return nullptr;
      }
      a->offset = start + size;
      a->live += 1;
      return a->base + start;
    }

    bool arena_free(void* ptr) {
      if (!graphs_enabled())
        return false;
      const char* p = static_cast<const char*>(ptr);
      const std::lock_guard<std::mutex> lock(mutex);
      for (auto& a : arenas) {
        if (a->base && p >= a->base && p < a->base + a->capacity) {
          if (--a->live == 0 && a->released)
            free_buffer(*a);
          return true;
        }
      }
      return false;
    }

    void note_allocation(void* ptr) {
      if (!step_capturing)
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

    void release_arenas() {
      const std::lock_guard<std::mutex> lock(mutex);
      for (Arena* a : own) {
        if (!a)
          continue;
        a->released = true;
        a->peak = 0;
        if (a->live == 0)
          free_buffer(*a);
      }
    }

  }
}

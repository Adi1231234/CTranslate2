#include "cuda/graph_memory.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

#include "cuda/graph.h"

namespace ctranslate2 {
  namespace cuda {

    struct Arena {
      char* base = nullptr;
      size_t capacity = 0, offset = 0, peak = 0;
      long long live = 0;                     // allocations not freed yet
      bool released = false;                  // the decoding loop ended: free the buffer once empty
      bool overflowed = false;                // in this step: the next arena is made larger
      bool segment_overflowed = false;        // in the current captured segment: its graph has memory nodes
      cudaStream_t stream = nullptr;          // the owner's stream, which orders the buffer's own release
    };

    // Arena memory can be freed by any thread, so the record of the arenas is shared.
    static std::mutex mutex;
    static std::vector<std::unique_ptr<Arena>> arenas;           // never shrinks: pointers stay valid
    static thread_local Arena* own[2] = {nullptr, nullptr};
    static thread_local Arena* capturing = nullptr;
    static thread_local bool step_capturing = false;

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
      // A step's temporaries (the caches stay in the pool, CaptureBreak): the larger of the last two peaks and a
      // margin for the step to grow.
      const size_t want = std::max(a.peak, other.peak) + (size_t(64) << 20);
      if (a.capacity < want || a.overflowed) {
        free_buffer(a);
        a.stream = stream;
        a.capacity = want;
        // The buffer ends 1 MB after the last byte handed out, as a pool allocation is followed by more pool.
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&a.base), a.capacity + (size_t(1) << 20), stream));
        a.overflowed = false;
      }
      a.offset = 0;
      a.peak = 0;
      a.stream = stream;
      capturing = &a;
      step_capturing = true;
      return true;
    }

    static thread_local Arena* paused = nullptr;

    bool segment_memory_nodes() {
      Arena* a = capturing ? capturing : paused;
      const bool overflowed = a && a->segment_overflowed;
      if (a)
        a->segment_overflowed = false;
      return overflowed;
    }

    void end_step_arena() {
      capturing = paused = nullptr;
      step_capturing = false;
    }

    void pause_step_arena() {
      paused = capturing;
      capturing = nullptr;
      step_capturing = false;
    }

    void resume_step_arena() {
      capturing = paused;
      paused = nullptr;
      step_capturing = capturing != nullptr;
    }

    void* arena_allocate(size_t size) {
      Arena* a = capturing;
      if (!a)
        return nullptr;
      const std::lock_guard<std::mutex> lock(mutex);
      const size_t start = (a->offset + 255) / 256 * 256;
      a->peak = std::max(a->peak, start + size);
      if (start + size > a->capacity) {
        a->overflowed = a->segment_overflowed = true;             // the pool serves it; the next arena is larger
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

    bool capturing_step() {
      return step_capturing;
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

#include "cuda/persistent.h"

#include <unordered_map>

#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    int sm_count() {
      return get_device_properties().multiProcessorCount;
    }

    unsigned* work_counter(cudaStream_t stream) {
      // 4 bytes per thread and stream, like the thread's cuBLAS handle; never freed (a free at thread exit can
      // run after the CUDA context is gone).
      static thread_local std::unordered_map<cudaStream_t, unsigned*> counters;
      unsigned*& counter = counters[stream];
      if (!counter) {
        CUDA_CHECK(cudaMalloc(&counter, sizeof (unsigned)));
        CUDA_CHECK(cudaMemset(counter, 0, sizeof (unsigned)));
      }
      return counter;
    }

    int persistent_blocks_per_sm(const char* name) {
      return use_stock_kernels() ? 0 : read_int_from_env(name, 0);
    }

  }
}

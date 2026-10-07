#pragma once

#include <cuda_runtime.h>

namespace ctranslate2 {
  namespace cuda {

    // CT2_CUDA_POOL_REPORT_S=<n> (measurement): every n seconds, to stderr, the memory of the device's
    // stream-ordered pool in use and held (each with its high since the last report) and the device's free memory,
    // to tell what a run needs from what the pool keeps. Nothing without it.
    void start_pool_report(int device, cudaMemPool_t pool, const char* name);   // name: the lines' first word

  }
}

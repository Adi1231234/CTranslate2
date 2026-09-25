#pragma once

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // CT2_CUDA_GRAPHS=1: a decoding step's decoder work (~570 kernels for Whisper large-v3) is captured from the
    // thread's stream into a CUDA graph and launched as that one graph, made fresh each step (the cache grows).
    // Only how the kernels are launched changes, never what they compute. On the store PC's RTX 5060 Ti (WDDM) a
    // plain launch costs ~5 us of GPU time between dependent kernels, a graph node well under 1 us
    // (tools/turing/kernels/pdl_bench.cu: a 570-kernel step 5.1 ms plain, 3.1 ms as a fresh graph).
    // Streams are then all created ones: the legacy default stream cannot be captured.
    bool graphs_enabled();

    // Captures the thread's stream from construction to launch(); without `capture`, graphs or a capturable
    // stream it does nothing and the work runs as launched.
    class StepGraph {
    public:
      explicit StepGraph(bool capture);
      ~StepGraph();
      void launch();
    private:
      cudaStream_t _stream = nullptr;
      bool _capturing = false;
    };

  }
}

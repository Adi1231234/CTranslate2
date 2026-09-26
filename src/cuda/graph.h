#pragma once

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // CT2_CUDA_GRAPHS=1: a decoding step's decoder work (~570 kernels for Whisper large-v3) is captured from the
    // thread's stream and launched as one CUDA graph. The step allocates from a step arena (graph_memory.h), so the
    // graph has no memory nodes and the thread's executable graph is updated in place from step to step
    // (cudaGraphExecUpdate: the caches grow and move, the kernels stay), instantiated again only when the kernels
    // change. Only how the kernels are launched changes, never what they compute. On the store PC's RTX 5060 Ti
    // (WDDM) a plain launch costs ~5 us of GPU time between dependent kernels, a graph node ~1.4 us
    // (tools/turing/kernels/pdl_bench.cu). Streams are then all created ones: the legacy stream cannot be captured.
    // CT2_CUDA_GRAPHS_STATS=1 prints, after each decoding loop, how its steps ran.
    bool graphs_enabled();

    // Captures the thread's stream from construction to launch() for decoding step `step`; without graphs, a
    // capturable stream or a free step arena it does nothing and the work runs as launched.
    class StepGraph {
    public:
      explicit StepGraph(long long step);
      ~StepGraph();
      void launch();
    private:
      cudaStream_t _stream = nullptr;
      bool _capturing = false;
    };

    // Held by a decoding loop: at its end the thread's executable graph and step arenas are released.
    class StepGraphScope {
    public:
      StepGraphScope() = default;
      ~StepGraphScope();
      StepGraphScope(const StepGraphScope&) = delete;
      StepGraphScope& operator=(const StepGraphScope&) = delete;
    };

  }
}

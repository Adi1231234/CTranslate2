#pragma once

#include <cstddef>

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // CT2_CUDA_GRAPHS=1: a decoding step's decoder work (~570 kernels for Whisper large-v3) is captured from the
    // thread's stream and launched as CUDA graphs, one per segment: the attention cache updates (CaptureBreak) run
    // between segments as launched, so their large, growing allocations stay in the pool. A segment allocates from
    // the step arena (graph_memory.h), so its graph has no memory nodes and the executable graph of the same segment
    // of the previous step is updated in place (cudaGraphExecUpdate), instantiated again only when its kernels
    // change. Only how the kernels are launched changes, never what they compute. On the store PC's RTX 5060 Ti
    // (WDDM) a plain launch costs ~5 us of GPU time between dependent kernels, a graph node ~1.4 us
    // (tools/turing/kernels/pdl_bench.cu). Streams are then all created ones: the legacy stream cannot be captured.
    // CT2_CUDA_GRAPHS_STATS=1 prints, after each decoding loop, how its segments ran.
    bool graphs_enabled();

    // Captures the thread's stream from construction to launch() for decoding step `step`; without graphs, a
    // capturable stream or a free step arena it does nothing and the work runs as launched.
    class StepGraph {
    public:
      explicit StepGraph(long long step);
      ~StepGraph();
      void launch();
    private:
      friend class CaptureBreak;
      void begin_segment();
      void end_segment();
      cudaStream_t _stream = nullptr;
      bool _capturing = false;
      size_t _segment = 0;
    };

    // Inside a captured step: launches the segment captured so far, and resumes capturing when it goes out of
    // scope; the work in between runs as launched. Nothing outside a captured step.
    class CaptureBreak {
    public:
      CaptureBreak();
      ~CaptureBreak();
      CaptureBreak(const CaptureBreak&) = delete;
      CaptureBreak& operator=(const CaptureBreak&) = delete;
    private:
      StepGraph* _step = nullptr;
    };

    // Held by a decoding loop: at its end the thread's executable graphs and step arenas are released.
    class StepGraphScope {
    public:
      StepGraphScope() = default;
      ~StepGraphScope();
      StepGraphScope(const StepGraphScope&) = delete;
      StepGraphScope& operator=(const StepGraphScope&) = delete;
    };

  }
}

#include "cuda/graph.h"

#include <cstdio>

#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    bool graphs_enabled() {
      static const bool enabled = read_bool_from_env("CT2_CUDA_GRAPHS") && !use_stock_kernels();
      return enabled;
    }

    void report_failure(const char* call, const char* file, int line, const char* error) {
      if (!graphs_enabled())
        return;
      cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
      cudaStreamIsCapturing(get_cuda_stream(), &status);
      std::fprintf(stderr, "%s:%d: %s failed: %s (capture status %d)\n", file, line, call, error, int(status));
      std::fflush(stderr);
    }

    StepGraph::StepGraph(bool capture) {
      if (!capture || !graphs_enabled())
        return;
      _stream = get_cuda_stream();
      if (_stream == cudaStreamDefault)                 // the legacy default stream cannot be captured
        return;
      // Thread-local: other threads' work (the encoder beside the decoder) is not captured nor restricted.
      CUDA_CHECK(cudaStreamBeginCapture(_stream, cudaStreamCaptureModeThreadLocal));
      _capturing = true;
    }

    void StepGraph::launch() {
      if (!_capturing)
        return;
      _capturing = false;
      cudaGraph_t graph = nullptr;
      CUDA_CHECK(cudaStreamEndCapture(_stream, &graph));
      cudaGraphExec_t exec = nullptr;
      CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
      CUDA_CHECK(cudaGraphLaunch(exec, _stream));
      CUDA_CHECK(cudaGraphExecDestroy(exec));           // the launched work completes regardless
      CUDA_CHECK(cudaGraphDestroy(graph));
    }

    StepGraph::~StepGraph() {
      if (_capturing) {                                 // an exception left the step: end the capture
        cudaGraph_t graph = nullptr;
        cudaStreamEndCapture(_stream, &graph);
        if (graph)
          cudaGraphDestroy(graph);
      }
    }

  }
}

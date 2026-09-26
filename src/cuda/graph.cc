#include "cuda/graph.h"

#include <algorithm>
#include <cstdio>

#include "cuda/graph_memory.h"
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

    // The thread's executable graph, kept between steps and updated in place.
    static thread_local cudaGraphExec_t step_exec = nullptr;
    static thread_local long long steps_plain = 0, steps_updated = 0, steps_instantiated = 0, steps_overflowed = 0;
    static thread_local long long update_failures[16] = {};   // by cudaGraphExecUpdateResult

    StepGraph::StepGraph(long long step) {
      if (!graphs_enabled())
        return;
      _stream = get_cuda_stream();
      if (_stream == cudaStreamDefault || !begin_step_arena(step, _stream)) {   // legacy stream, or arena in use
        steps_plain += 1;
        return;
      }
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
      const bool memory_nodes = end_step_arena();
      if (step_exec && !memory_nodes) {
        cudaGraphExecUpdateResultInfo info;
        if (cudaGraphExecUpdate(step_exec, graph, &info) == cudaSuccess) {
          steps_updated += 1;
        } else {
          update_failures[std::min<int>(int(info.result), 15)] += 1;
          (void)cudaGetLastError();                     // another topology: instantiate below
          CUDA_CHECK(cudaGraphExecDestroy(step_exec));
          step_exec = nullptr;
        }
      }
      if (memory_nodes && step_exec) {                  // a graph with memory nodes is never kept
        CUDA_CHECK(cudaGraphExecDestroy(step_exec));
        step_exec = nullptr;
      }
      if (!step_exec) {
        CUDA_CHECK(cudaGraphInstantiate(&step_exec, graph, 0));
        steps_instantiated += 1;
      }
      CUDA_CHECK(cudaGraphLaunch(step_exec, _stream));
      if (memory_nodes) {
        steps_overflowed += 1;
        CUDA_CHECK(cudaGraphExecDestroy(step_exec));    // the launched work completes regardless
        step_exec = nullptr;
      }
      CUDA_CHECK(cudaGraphDestroy(graph));
      release_deferred(_stream);
    }

    StepGraph::~StepGraph() {
      if (_capturing) {                                 // an exception left the step: end the capture
        cudaGraph_t graph = nullptr;
        cudaStreamEndCapture(_stream, &graph);
        end_step_arena();
        if (graph)
          cudaGraphDestroy(graph);
        release_deferred(_stream);
      }
    }

    StepGraphScope::~StepGraphScope() {
      if (!graphs_enabled())
        return;
      if (step_exec)
        cudaGraphExecDestroy(step_exec);
      step_exec = nullptr;
      release_arenas();
      static const bool stats = read_bool_from_env("CT2_CUDA_GRAPHS_STATS");
      if (stats) {
        std::fprintf(stderr, "cuda graphs: %lld updated, %lld instantiated (%lld with memory nodes), %lld plain;"
                     " update failures by cudaGraphExecUpdateResult:", steps_updated, steps_instantiated,
                     steps_overflowed, steps_plain);
        for (int r = 0; r < 16; ++r)
          if (update_failures[r])
            std::fprintf(stderr, " %d x%lld", r, update_failures[r]);
        std::fprintf(stderr, "\n");
      }
      steps_plain = steps_updated = steps_instantiated = steps_overflowed = 0;
      for (long long& n : update_failures)
        n = 0;
    }

  }
}

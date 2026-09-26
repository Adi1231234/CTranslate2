#include "cuda/graph.h"

#include <cstdio>
#include <vector>

#include "cuda/graph_exec.h"
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

    static thread_local std::vector<cudaGraphExec_t> execs;   // per segment of a step, kept between steps
    static thread_local StepGraph* active = nullptr;

    StepGraph::StepGraph(long long step) {
      if (!graphs_enabled())
        return;
      _stream = get_cuda_stream();
      if (_stream == cudaStreamDefault || !begin_step_arena(step, _stream)) {   // legacy stream, or arena in use
        count_plain_step();
        return;
      }
      begin_segment();
      active = this;
    }

    void StepGraph::begin_segment() {
      // Thread-local: other threads' work (the encoder beside the decoder) is not captured nor restricted.
      CUDA_CHECK(cudaStreamBeginCapture(_stream, cudaStreamCaptureModeThreadLocal));
      _capturing = true;
    }

    void StepGraph::end_segment() {
      _capturing = false;
      cudaGraph_t graph = nullptr;
      CUDA_CHECK(cudaStreamEndCapture(_stream, &graph));
      if (execs.size() <= _segment)
        execs.resize(_segment + 1, nullptr);
      launch_segment(graph, execs[_segment], segment_memory_nodes(), _stream, _segment);
      _segment += 1;
      CUDA_CHECK(cudaGraphDestroy(graph));
      release_deferred(_stream);
    }

    void StepGraph::launch() {
      if (!_capturing)
        return;
      end_segment();
      end_step_arena();
      active = nullptr;
    }

    StepGraph::~StepGraph() {
      if (_capturing) {                                 // an exception left the step: end the capture
        _capturing = false;
        cudaGraph_t graph = nullptr;
        cudaStreamEndCapture(_stream, &graph);
        if (graph)
          cudaGraphDestroy(graph);
        end_step_arena();
        release_deferred(_stream);
      }
      if (active == this)
        active = nullptr;
    }

    CaptureBreak::CaptureBreak() {
      if (!active || !active->_capturing)
        return;
      _step = active;
      _step->end_segment();
      pause_step_arena();
    }

    CaptureBreak::~CaptureBreak() {
      if (!_step)
        return;
      resume_step_arena();
      if (cudaStreamBeginCapture(_step->_stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess)
        _step->_capturing = true;                       // else the rest of the step runs as launched
    }

    StepGraphScope::~StepGraphScope() {
      if (!graphs_enabled())
        return;
      for (cudaGraphExec_t exec : execs)
        if (exec)
          cudaGraphExecDestroy(exec);
      execs.clear();
      release_arenas();
      report_graph_stats();
    }

  }
}

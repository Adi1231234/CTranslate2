#include "cuda/graph.h"

#include <cstdio>
#include <cstdlib>
#include <exception>

#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    // A CUDA error inside a capture can end in std::terminate (a destructor's free failing while the error
    // unwinds); print what terminated the process first.
    static void report_terminate() {
      if (std::exception_ptr e = std::current_exception()) {
        try {
          std::rethrow_exception(e);
        } catch (const std::exception& x) {
          std::fprintf(stderr, "terminate during a CUDA graph step: %s\n", x.what());
        } catch (...) {
          std::fprintf(stderr, "terminate during a CUDA graph step: unknown exception\n");
        }
      }
      std::fflush(stderr);
      std::abort();
    }

    bool graphs_enabled() {
      static const bool enabled = [] {
        const bool on = read_bool_from_env("CT2_CUDA_GRAPHS") && !use_stock_kernels();
        if (on)
          std::set_terminate(report_terminate);
        return on;
      }();
      return enabled;
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

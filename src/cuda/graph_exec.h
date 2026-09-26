#pragma once

#include <cstddef>

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // Launches a captured segment's graph on `stream` through `exec`, the executable graph of the same segment of
    // the previous step: updated in place when the kernels are the same (cudaGraphExecUpdate), else instantiated
    // again. A graph with memory nodes is launched once and not kept.
    void launch_segment(cudaGraph_t graph, cudaGraphExec_t& exec, bool memory_nodes, cudaStream_t stream,
                        size_t segment);
    void count_plain_step();

    // CT2_CUDA_GRAPHS_STATS=1: prints how the thread's segments ran since the last call; resets the counts.
    void report_graph_stats();

  }
}

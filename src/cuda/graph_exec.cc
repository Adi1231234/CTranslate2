#include "cuda/graph_exec.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    static thread_local long long steps_plain = 0, updated = 0, instantiated = 0;
    static thread_local long long with_memory_nodes = 0, with_clusters = 0;
    static thread_local long long update_failures[16] = {};   // by cudaGraphExecUpdateResult

    static bool stats() {
      static const bool on = read_bool_from_env("CT2_CUDA_GRAPHS_STATS");
      return on;
    }

    static bool update(cudaGraphExec_t exec, cudaGraph_t graph, size_t segment) {
      cudaGraphExecUpdateResultInfo info;
      const cudaError_t e = cudaGraphExecUpdate(exec, graph, &info);
      if (e == cudaSuccess)
        return true;
      static bool reported = false;                     // the first failure, for the stats
      if (!reported && stats()) {
        cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
        if (info.errorNode)
          cudaGraphNodeGetType(info.errorNode, &type);
        std::fprintf(stderr, "cuda graphs: segment %zu update failed: %s (result %d, node type %d)\n", segment,
                     cudaGetErrorString(e), int(info.result), int(type));
        reported = true;
      }
      update_failures[std::min<int>(int(info.result), 15)] += 1;
      (void)cudaGetLastError();
      return false;
    }

    // cudaGraphExecUpdate does not carry thread block cluster dimensions over correctly (MLX #2813 instantiates
    // such graphs again instead): true when a kernel node of the graph launches clusters.
    static bool has_clusters(cudaGraph_t graph) {
      size_t n = 0;
      CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &n));
      std::vector<cudaGraphNode_t> nodes(n);
      CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &n));
      for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        cudaLaunchAttributeValue value = {};
        if (type == cudaGraphNodeTypeKernel
            && cudaGraphKernelNodeGetAttribute(node, cudaLaunchAttributeClusterDimension, &value) == cudaSuccess
            && value.clusterDim.x * value.clusterDim.y * value.clusterDim.z > 1)
          return true;
      }
      return false;
    }

    void launch_segment(cudaGraph_t graph, cudaGraphExec_t& exec, bool memory_nodes, cudaStream_t stream,
                        size_t segment) {
      memsets_to_kernels(graph);
      if (has_clusters(graph)) {
        with_clusters += 1;
        memory_nodes = true;                            // launched once from a fresh instance, as memory nodes are
      }
      if (exec && !memory_nodes && update(exec, graph, segment)) {
        updated += 1;
      } else if (exec) {                                // another topology, or memory nodes: a new instance
        CUDA_CHECK(cudaGraphExecDestroy(exec));
        exec = nullptr;
      }
      if (!exec) {
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        instantiated += 1;
      }
      CUDA_CHECK(cudaGraphLaunch(exec, stream));
      if (memory_nodes) {
        with_memory_nodes += 1;
        CUDA_CHECK(cudaGraphExecDestroy(exec));         // the launched work completes regardless
        exec = nullptr;
      }
    }

    void count_plain_step() {
      steps_plain += 1;
    }

    void report_graph_stats() {
      if (stats()) {
        std::fprintf(stderr, "cuda graphs: segments %lld updated, %lld instantiated (%lld with memory nodes); "
                     "%lld with clusters; %lld steps plain; update failures by cudaGraphExecUpdateResult:", updated,
                     instantiated, with_memory_nodes, with_clusters, steps_plain);
        for (int r = 0; r < 16; ++r)
          if (update_failures[r])
            std::fprintf(stderr, " %d x%lld", r, update_failures[r]);
        std::fprintf(stderr, "\n");
      }
      steps_plain = updated = instantiated = with_memory_nodes = with_clusters = 0;
      for (long long& n : update_failures)
        n = 0;
    }

  }
}

#ifndef CT2_USE_HIP

#include "cuda/graph_exec.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ctranslate2 {
  namespace cuda {

    // A memset node's work as a kernel: `value` (its low `element_size` bytes) in `width` elements of each of
    // `height` rows, `pitch` bytes apart. Byte for byte what the memset writes.
    __global__ void graph_memset_kernel(char* dst, size_t pitch, unsigned value, unsigned element_size,
                                        size_t width, size_t height) {
      const size_t count = width * height;
      for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        char* p = dst + (i / width) * pitch + (i % width) * element_size;
        if (element_size == 4)
          *reinterpret_cast<uint32_t*>(p) = value;
        else if (element_size == 2)
          *reinterpret_cast<uint16_t*>(p) = static_cast<uint16_t>(value);
        else
          *p = static_cast<char>(value);
      }
    }

    void memsets_to_kernels(cudaGraph_t graph) {
      size_t n = 0;
      CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &n));
      std::vector<cudaGraphNode_t> nodes(n);
      CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &n));
      for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeMemset)
          continue;
        cudaMemsetParams m;
        CUDA_CHECK(cudaGraphMemsetNodeGetParams(node, &m));
        size_t in = 0, out = 0;
        CUDA_CHECK(cudaGraphNodeGetDependencies(node, nullptr, &in));
        CUDA_CHECK(cudaGraphNodeGetDependentNodes(node, nullptr, &out));
        std::vector<cudaGraphNode_t> before(in), after(out);
        CUDA_CHECK(cudaGraphNodeGetDependencies(node, before.data(), &in));
        CUDA_CHECK(cudaGraphNodeGetDependentNodes(node, after.data(), &out));
        char* dst = static_cast<char*>(m.dst);
        size_t pitch = m.pitch, width = m.width, height = m.height;
        unsigned value = m.value, element_size = m.elementSize;
        void* args[] = {&dst, &pitch, &value, &element_size, &width, &height};
        cudaKernelNodeParams k = {};
        k.func = reinterpret_cast<void*>(&graph_memset_kernel);
        k.blockDim = dim3(256);
        k.gridDim = dim3(static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>((width * height + 255) / 256, 1024))));
        k.kernelParams = args;
        cudaGraphNode_t kernel;
        CUDA_CHECK(cudaGraphAddKernelNode(&kernel, graph, before.data(), in, &k));
        for (cudaGraphNode_t next : after)
          CUDA_CHECK(cudaGraphAddDependencies(graph, &kernel, &next, 1));
        CUDA_CHECK(cudaGraphDestroyNode(node));
      }
    }

  }
}

#endif

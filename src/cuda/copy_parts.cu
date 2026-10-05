#include "cuda/copy_parts.h"

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // One thread per 16-byte vector of out; its part from the parts' cumulative ends.
    __global__ void copy_parts_kernel(CopyParts parts, uint4* out, size_t total) {
      const size_t v = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
      if (v >= total)
        return;
      size_t begin = 0;
      int p = 0;
      while (v >= begin + parts.bytes[p] / 16)
        begin += parts.bytes[p++] / 16;
      out[v] = static_cast<const uint4*>(parts.src[p])[v - begin];
    }

    bool copy_parts_supported(const void* p, size_t bytes) {
      return reinterpret_cast<uintptr_t>(p) % 16 == 0 && bytes % 16 == 0;
    }

    void copy_parts(const CopyParts& parts, void* out) {
      size_t total = 0;
      for (int p = 0; p < parts.count; ++p)
        total += parts.bytes[p] / 16;
      if (total == 0)
        return;
      constexpr unsigned threads = 256;
      copy_parts_kernel<<<(total + threads - 1) / threads, threads, 0, get_cuda_stream()>>>(
        parts, static_cast<uint4*>(out), total);
    }

  }
}

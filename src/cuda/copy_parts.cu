#include "cuda/copy_parts.h"

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // One thread per 16-byte vector of out; its part from the parts' cumulative ends, picked in a loop over constant
    // indices (a parameter array indexed by a computed part would be copied to local memory by every thread).
    __global__ void copy_parts_kernel(CopyParts parts, uint4* out, size_t total) {
      const size_t v = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
      if (v >= total)
        return;
      size_t begin = 0, end = 0;
      const void* src = nullptr;
      #pragma unroll
      for (int q = 0; q < CopyParts::max_parts; ++q)
        if (q < parts.count) {
          const size_t next = end + parts.bytes[q] / 16;
          if (v >= end && v < next) {
            begin = end;
            src = parts.src[q];
          }
          end = next;
        }
      out[v] = static_cast<const uint4*>(src)[v - begin];
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

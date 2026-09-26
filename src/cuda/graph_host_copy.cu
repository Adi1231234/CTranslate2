#ifndef CT2_USE_HIP

#include "cuda/graph_host_copy.h"
#include "cuda/graph.h"

#include <algorithm>
#include <cstring>

namespace ctranslate2 {
  namespace cuda {

    // Host bytes carried as a kernel parameter: the parameters are copied when the launch is captured (and when
    // the graph is updated), so the copy does not read the host memory later, when the graph runs.
    struct HostChunk {
      unsigned char bytes[1024];
    };

    __global__ void host_chunk_kernel(unsigned char* dst, HostChunk chunk, unsigned n) {
      for (unsigned i = threadIdx.x; i < n; i += blockDim.x)
        dst[i] = chunk.bytes[i];
    }

    bool copy_host_bytes_if_capturing(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
      if (!graphs_enabled() || bytes == 0 || bytes > (size_t(64) << 10))
        return false;
      cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
      if (cudaStreamIsCapturing(stream, &status) != cudaSuccess || status != cudaStreamCaptureStatusActive)
        return false;
      HostChunk chunk;
      for (size_t done = 0; done < bytes; done += sizeof (chunk.bytes)) {
        const unsigned n = static_cast<unsigned>(std::min(sizeof (chunk.bytes), bytes - done));
        std::memcpy(chunk.bytes, static_cast<const unsigned char*>(src) + done, n);
        host_chunk_kernel<<<1, 256, 0, stream>>>(static_cast<unsigned char*>(dst) + done, chunk, n);
      }
      CUDA_CHECK(cudaGetLastError());
      return true;
    }

  }
}

#endif

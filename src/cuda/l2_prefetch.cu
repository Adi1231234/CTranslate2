#ifndef CT2_USE_HIP

#include "cuda/l2_prefetch.h"

#include <algorithm>

#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    constexpr int l2_max_ranges = 4, l2_line = 128;

    struct L2Ranges {
      const char* data[l2_max_ranges];
      size_t lines[l2_max_ranges];
      int count;
    };

    __global__ void l2_prefetch_kernel(L2Ranges r) {
      const size_t first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
      const size_t step = size_t(gridDim.x) * blockDim.x;
      for (int i = 0; i < r.count; ++i)
        for (size_t line = first; line < r.lines[i]; line += step)
          asm volatile("prefetch.global.L2 [%0];" :: "l"(r.data[i] + line * l2_line));
    }

    int l2_prefetch_count() {
      static const int count = use_stock_kernels() ? 0 : read_int_from_env("CT2_L2_PREFETCH", 0);
      return count;
    }

    void l2_prefetch(const std::vector<const StorageView*>& buffers) {
      L2Ranges r{};
      size_t most = 0;
      for (const StorageView* x : buffers) {
        if (r.count == l2_max_ranges || !x || x->device() != Device::CUDA)
          continue;
        r.data[r.count] = static_cast<const char*>(x->buffer());
        r.lines[r.count] = (x->size() * x->item_size() + l2_line - 1) / l2_line;
        most = std::max(most, r.lines[r.count]);
        ++r.count;
      }
      if (most == 0)
        return;
      constexpr unsigned threads = 256;
      const unsigned blocks = unsigned(std::min<size_t>((most + threads - 1) / threads, 1024));
      l2_prefetch_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(r);
    }

  }
}

#endif

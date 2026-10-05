#include "cuda/softmax_parts.h"

#include <algorithm>

#include "cuda/helpers.h"
#include "cuda/utils.h"
#include "ops/softmax_kernels.cuh"

namespace ctranslate2 {
  namespace cuda {

    // Each part's legacy block (the block the legacy kernel would launch for its rows, whose arithmetic the warp
    // kernel replays).
    struct SoftmaxBlocks {
      unsigned block[SoftmaxParts::max_parts];
    };

    // Warp w of block b takes row 4b + w of the parts' rows, as warp_softmax_forward does for one tensor. The parts'
    // fields are picked in loops over constant indices (a parameter array indexed by a computed part is copied to
    // local memory by every thread).
    __global__ void __launch_bounds__(at::native::warp_softmax_rows_per_block * C10_WARP_SIZE)
    softmax_parts_kernel(SoftmaxParts parts, SoftmaxBlocks blocks, unsigned rows, unsigned max_cols) {
      extern __shared__ float parts_smem[];
      const unsigned warp = threadIdx.x / C10_WARP_SIZE, lane = threadIdx.x % C10_WARP_SIZE;
      const unsigned row = blockIdx.x * at::native::warp_softmax_rows_per_block + warp;
      if (row >= rows)
        return;
      __half* data = nullptr;
      unsigned cols = 0, block = 0, begin = 0;
      #pragma unroll
      for (int q = 0; q < SoftmaxParts::max_parts; ++q)
        if (q < parts.count && row < parts.rows_end[q] && (q == 0 || row >= parts.rows_end[q - 1])) {
          data = static_cast<__half*>(parts.data[q]);
          cols = parts.cols[q];
          block = blocks.block[q];
          begin = q == 0 ? 0 : parts.rows_end[q - 1];
        }
      __half* x = data + size_t(row - begin) * cols;
      float* buf = parts_smem + warp * (at::native::warp_softmax_slot(max_cols) + 1);
      at::native::warp_softmax_row<__half, false>(x, x, cols, block, lane, buf);
    }

    bool softmax_parts_supported(dim_t cols) {
      return !use_stock_kernels() && cols > 0 && cols <= 1024;   // softmax_rows1024 takes rows past 1024
    }

    void softmax_parts(const SoftmaxParts& parts) {
      if (parts.count == 0)
        return;
      SoftmaxBlocks blocks{};
      unsigned max_cols = 0;
      for (int p = 0; p < parts.count; ++p) {
        blocks.block[p] = get_block_size(parts.cols[p]).x;
        max_cols = std::max(max_cols, parts.cols[p]);
      }
      const unsigned rows = parts.rows_end[parts.count - 1];
      constexpr unsigned per_block = at::native::warp_softmax_rows_per_block;
      const size_t smem = per_block * (at::native::warp_softmax_slot(max_cols) + 1) * sizeof (float);
      softmax_parts_kernel<<<(rows + per_block - 1) / per_block, per_block * C10_WARP_SIZE, smem,
                             get_cuda_stream()>>>(parts, blocks, rows, max_cols);
    }

  }
}

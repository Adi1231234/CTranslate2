#pragma once

// Several rows each with the bits cuBLAS gives a product of one row (a gemv: internal::gemvx::kernel), reading the
// weights once for them all. On the L40S with cuBLAS 12.9.2 (tools/turing/kernels/gemv_probe.cu, every output of
// 3 fills exact): y_j = half(sum_i W[j][i] x[i]) as T partial sums, partial t over i = t, t + T, ... in order, then
// a tree from the halves (s_t += s_(t + T/2), ... s_t += s_(t + 1)); T = 32 for 1280 x 1280, 8 for 51866 x 1280, 16
// for 3840, 5120 or 51872 x 1280 and 1280 x 5120. Products of halves are exact in fp32, so a fused multiply-add is
// the product's sum. A launch takes up to sr_max_rows rows; T threads an output, sr_threads / T outputs a block.

#include <cstdint>
#include <cuda_fp16.h>

namespace ctranslate2 {
  namespace cuda {

    constexpr int sr_max_rows = 16, sr_threads = 256;

    struct SingleRows {
      int count;
      const __half* x[sr_max_rows];                          // each row's input (k values)
      __half* y[sr_max_rows];                                // each row's output (n values)
    };

    template <int T>
    __global__ void single_rows_gemv_kernel(const __half* w, SingleRows rows, int n, int k) {
      const int t = threadIdx.x % T, j = blockIdx.x * (sr_threads / T) + threadIdx.x / T;
      const bool live = j < n;
      const __half* wj = w + static_cast<size_t>(live ? j : 0) * k;
      float s[sr_max_rows];
      #pragma unroll
      for (int r = 0; r < sr_max_rows; ++r)
        s[r] = 0.f;
      for (int i = t; i < k; i += T) {
        const float v = __half2float(__ldg(wj + i));
        #pragma unroll
        for (int r = 0; r < sr_max_rows; ++r)
          if (r < rows.count)
            s[r] = fmaf(v, __half2float(__ldg(rows.x[r] + i)), s[r]);
      }
      #pragma unroll
      for (int r = 0; r < sr_max_rows; ++r) {
        if (r >= rows.count)
          break;
        float sum = s[r];
        #pragma unroll
        for (int off = T / 2; off > 0; off /= 2)             // lane t < off: s_t + s_(t + off), the halves' tree
          sum += __shfl_down_sync(0xffffffffu, sum, off, T);
        if (t == 0 && live)
          rows.y[r][j] = __float2half_rn(sum);
      }
    }

    // T for a product of n x k (the recovered shapes), or 0.
    inline int single_rows_partials(int64_t n, int64_t k) {
      if (k == 1280)
        return n == 1280 ? 32 : n == 51866 ? 8 : (n == 3840 || n == 5120 || n == 51872) ? 16 : 0;
      return (k == 5120 && n == 1280) ? 16 : 0;
    }

  }
}

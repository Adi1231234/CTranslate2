#pragma once

// Several rows each with the bits cuBLAS gives a product of one row (a gemv: internal::gemvx::kernel), reading the
// weights once for them all. On the L40S with cuBLAS 12.9.2 (tools/turing/kernels/gemv_probe.cu, every output of
// 3 fills exact): y_j = half(sum_i W[j][i] x[i]) as T partial sums, partial t over i = t, t + T, ... in order, then
// a tree from the halves (s_t += s_(t + T/2), ... s_t += s_(t + 1)); T = 32 for 1280 x 1280, 8 for 51866 x 1280, 16
// for 3840, 5120 or 51872 x 1280 and 1280 x 5120. Products of halves are exact in fp32, so a fused multiply-add is
// the product's sum. T threads share O outputs, each thread their partial t for every row: an input value read once
// serves O outputs and a weight read once serves every row (long28: one output a group, a ladder call took 32 s
// instead of 25). O = 4, but 1 for k = 5120 (n = 1280 outputs of 16 threads are few enough already: with 4 a group,
// 16 rows took 199 us against 180 for 16 cuBLAS calls, srcheck2). A launch takes up to sr_max_rows rows.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ctranslate2 {
  namespace cuda {

    constexpr int sr_max_rows = 16, sr_threads = 256;

    struct SingleRows {
      int count;
      const __half* x[sr_max_rows];                          // each row's input (k values); past count, x[0]
      __half* y[sr_max_rows];                                // each row's output (n values)
    };

    template <int T, int O, int R>
    __device__ __forceinline__ void single_rows_outputs(const __half* w, const SingleRows& rows, int j0, int n, int k,
                                                        int t) {
      float s[R][O];
      #pragma unroll
      for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int o = 0; o < O; ++o)
          s[r][o] = 0.f;
      const __half* wo[O];
      #pragma unroll
      for (int o = 0; o < O; ++o)
        wo[o] = w + static_cast<size_t>(min(j0 + o, n - 1)) * k;
      for (int i = t; i < k; i += T) {
        float v[O];
        #pragma unroll
        for (int o = 0; o < O; ++o)
          v[o] = __half2float(__ldg(wo[o] + i));
        #pragma unroll
        for (int r = 0; r < R; ++r) {
          const float x = __half2float(__ldg(rows.x[r] + i));
          #pragma unroll
          for (int o = 0; o < O; ++o)
            s[r][o] = fmaf(v[o], x, s[r][o]);
        }
      }
      #pragma unroll
      for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int o = 0; o < O; ++o) {
          float sum = s[r][o];
          #pragma unroll
          for (int off = T / 2; off > 0; off /= 2)           // lane t < off: s_t + s_(t + off), the halves' tree
            sum += __shfl_down_sync(0xffffffffu, sum, off, T);
          if (t == 0 && j0 + o < n && r < rows.count)
            rows.y[r][j0 + o] = __float2half_rn(sum);
        }
    }

    // The rows' count picks R (the registers of R x O sums); every thread of the block takes part.
    template <int T, int O>
    __global__ void single_rows_gemv_kernel(const __half* w, SingleRows rows, int n, int k) {
      const int t = threadIdx.x % T;
      const int j0 = (blockIdx.x * (sr_threads / T) + threadIdx.x / T) * O;
      switch ((rows.count + 3) / 4) {
      case 1: single_rows_outputs<T, O, 4>(w, rows, j0, n, k, t); break;
      case 2: single_rows_outputs<T, O, 8>(w, rows, j0, n, k, t); break;
      case 3: single_rows_outputs<T, O, 12>(w, rows, j0, n, k, t); break;
      default: single_rows_outputs<T, O, 16>(w, rows, j0, n, k, t);
      }
    }

    // T for a product of n x k (the recovered shapes), or 0.
    inline int single_rows_partials(int64_t n, int64_t k) {
      if (k == 1280)
        return n == 1280 ? 32 : n == 51866 ? 8 : (n == 3840 || n == 5120 || n == 51872) ? 16 : 0;
      return (k == 5120 && n == 1280) ? 16 : 0;
    }

    // Outputs a group of T threads for a product of k columns, and the launch's blocks for n outputs.
    inline int single_rows_outputs_of(int k) {
      return k >= 5120 ? 1 : 4;
    }
    inline int single_rows_blocks(int n, int T, int O) {
      const int per_block = (sr_threads / T) * O;
      return (n + per_block - 1) / per_block;
    }

    // The launch for T and O (single_rows_partials, single_rows_outputs_of).
    inline void single_rows_launch(const __half* w, const SingleRows& rows, int n, int k, int T, int O,
                                   cudaStream_t stream = 0) {
      const int blocks = single_rows_blocks(n, T, O);
      if (O == 1) {
        if (T == 32) single_rows_gemv_kernel<32, 1><<<blocks, sr_threads, 0, stream>>>(w, rows, n, k);
        else if (T == 16) single_rows_gemv_kernel<16, 1><<<blocks, sr_threads, 0, stream>>>(w, rows, n, k);
        else single_rows_gemv_kernel<8, 1><<<blocks, sr_threads, 0, stream>>>(w, rows, n, k);
      } else {
        if (T == 32) single_rows_gemv_kernel<32, 4><<<blocks, sr_threads, 0, stream>>>(w, rows, n, k);
        else if (T == 16) single_rows_gemv_kernel<16, 4><<<blocks, sr_threads, 0, stream>>>(w, rows, n, k);
        else single_rows_gemv_kernel<8, 4><<<blocks, sr_threads, 0, stream>>>(w, rows, n, k);
      }
    }

  }
}

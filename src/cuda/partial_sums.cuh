#pragma once

// Sums with the arithmetic of a cuBLAS gemv (the replicas of slot_attention.cuh and ladder_cross.cu): T partial sums,
// each over its own elements in increasing order (strided: (r + T j) W + u, W consecutive elements a turn; or
// contiguous: one chunk of ceil(n / T)), combined by a tree from the halves (tree 1: s_r += s_(r + T/2) ..
// s_r += s_(r + 1)), from the neighbours (tree 2) or in order (tree 0). Products of halves are exact in fp32, so a
// fused multiply-add is the product's sum.

#include <cuda_fp16.h>

namespace ctranslate2 {
  namespace cuda {

    template <int T>
    static __device__ __forceinline__ float combine(float* s, int tree) {
      if (tree == 1) {
        #pragma unroll
        for (int off = T / 2; off > 0; off /= 2)
          #pragma unroll
          for (int r = 0; r < off; ++r)
            s[r] += s[r + off];
        return s[0];
      }
      if (tree == 2) {
        #pragma unroll
        for (int off = 1; off < T; off *= 2)
          #pragma unroll
          for (int r = 0; r + off < T; r += 2 * off)
            s[r] += s[r + off];
        return s[0];
      }
      float sum = 0.f;
      #pragma unroll
      for (int r = 0; r < T; ++r)
        sum += s[r];
      return sum;
    }

    constexpr int split_lanes = 8;   // the block's y lanes sharing an output's partials

    // Partial r of n elements, a(i) b(i): its own elements in increasing order from zero, as one thread of a gemv sums
    // them (strided: (r + T j) W + u; or contiguous: one chunk of ceil(n / T)), each with a fused multiply-add.
    template <int T, int W, bool CONTIGUOUS, typename A, typename B>
    static __device__ __forceinline__ float partial_sum(int r, int n, const A& a, const B& b) {
      float s = 0.f;
      if (CONTIGUOUS) {
        const int chunk = (n + T - 1) / T, end = min(n, (r + 1) * chunk);
        for (int i = r * chunk; i < end; ++i)
          s = fmaf(a(i), b(i), s);
      } else {
        for (int base = r * W; base < n; base += T * W)
          #pragma unroll
          for (int u = 0; u < W; ++u)
            if (base + u < n)
              s = fmaf(a(base + u), b(base + u), s);
      }
      return s;
    }

    // partial_sum of two columns at once, a(i) b(i).x and a(i) b(i).y (b(i) a float2, e.g. a pair of halves read
    // together): each column's sum is partial_sum's, element by element.
    template <int T, int W, bool CONTIGUOUS, typename A, typename B>
    static __device__ __forceinline__ float2 partial_sum2(int r, int n, const A& a, const B& b) {
      float2 s = make_float2(0.f, 0.f);
      const auto add = [&](int i) {
        const float x = a(i);
        const float2 y = b(i);
        s.x = fmaf(x, y.x, s.x);
        s.y = fmaf(x, y.y, s.y);
      };
      if (CONTIGUOUS) {
        const int chunk = (n + T - 1) / T, end = min(n, (r + 1) * chunk);
        for (int i = r * chunk; i < end; ++i)
          add(i);
      } else {
        for (int base = r * W; base < n; base += T * W)
          #pragma unroll
          for (int u = 0; u < W; ++u)
            if (base + u < n)
              add(base + u);
      }
      return s;
    }

    // One output column x (of a block's 32) over n elements, a(i) b(i): the T partials spread over `lanes` lanes
    // (lane y sums partials y, y + lanes, ...) into sm ([T][32] floats); then lane 0 combines them. Every thread of
    // the block calls it (it synchronizes); the sum is lane 0's.
    template <int T, int W, bool CONTIGUOUS, int TREE, typename A, typename B>
    static __device__ __forceinline__ float split_partials(float (*sm)[32], int x, int y, int n, const A& a,
                                                           const B& b, int lanes = split_lanes) {
      for (int r = y; r < T; r += lanes)
        sm[r][x] = partial_sum<T, W, CONTIGUOUS>(r, n, a, b);
      __syncthreads();
      float sum = 0.f;
      if (y == 0) {
        float s[T];
        #pragma unroll
        for (int r = 0; r < T; ++r)
          s[r] = sm[r][x];
        sum = combine<T>(s, TREE);
      }
      return sum;
    }

  }
}

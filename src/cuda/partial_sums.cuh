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

    // One output column x (of a block's 32) over n elements, a(i) b(i): the T partials spread over the block's
    // split_lanes y lanes (lane y sums partials y, y + split_lanes, ...), each as one thread would sum it, into sm
    // ([T][32] floats); then lane 0 combines them. Every thread of the block calls it (it synchronizes); the sum is
    // lane 0's.
    template <int T, int W, bool CONTIGUOUS, int TREE, typename A, typename B>
    static __device__ __forceinline__ float split_partials(float (*sm)[32], int x, int y, int n, const A& a,
                                                           const B& b) {
      for (int r = y; r < T; r += split_lanes) {
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
        sm[r][x] = s;
      }
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

    // split_partials for `rows` (at most R) sums over the same b(i), a(k, i) for sum k: every sum's arithmetic is
    // split_partials' (the same elements in the same order, the same tree), each b(i) read once for all of them.
    // sm: rows x T x 32 floats. Lane 0 hands each sum k to store(k, sum).
    template <int T, int W, bool CONTIGUOUS, int TREE, int R, typename A, typename B, typename S>
    static __device__ __forceinline__ void split_partials_rows(float* sm, int x, int y, int n, int rows, const A& a,
                                                               const B& b, const S& store) {
      for (int r = y; r < T; r += split_lanes) {
        float s[R];
        #pragma unroll
        for (int k = 0; k < R; ++k)
          s[k] = 0.f;
        const auto add = [&](int i) {
          const float v = b(i);
          #pragma unroll
          for (int k = 0; k < R; ++k)
            if (k < rows)
              s[k] = fmaf(a(k, i), v, s[k]);
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
        #pragma unroll
        for (int k = 0; k < R; ++k)
          if (k < rows)
            sm[(k * T + r) * 32 + x] = s[k];
      }
      __syncthreads();
      if (y == 0)
        for (int k = 0; k < rows; ++k) {
          float s[T];
          #pragma unroll
          for (int r = 0; r < T; ++r)
            s[r] = sm[(k * T + r) * 32 + x];
          store(k, combine<T>(s, TREE));
        }
    }

  }
}

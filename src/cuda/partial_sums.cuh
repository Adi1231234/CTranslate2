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

    // Partials first, first + step, ... (Q of them, those below T) of n elements, each over its own elements in
    // increasing order (strided: (r + T j) W + u; or contiguous: one chunk of ceil(n / T)) as one thread of a gemv
    // sums them: f(k, i) for partial k's element i, the Q partials' turns interleaved (independent chains for the
    // scheduler), each partial's in its own order.
    template <int T, int W, bool CONTIGUOUS, int Q, typename F>
    static __device__ __forceinline__ void for_partials(int first, int step, int n, const F& f) {
      if (CONTIGUOUS) {
        const int chunk = (n + T - 1) / T;
        for (int o = 0; o < chunk; ++o)
          #pragma unroll
          for (int k = 0; k < Q; ++k) {
            const int r = first + k * step, i = r * chunk + o;
            if (r < T && i < n)
              f(k, i);
          }
      } else {
        for (int base = 0; base < n; base += T * W)
          #pragma unroll
          for (int k = 0; k < Q; ++k) {
            const int r = first + k * step;
            #pragma unroll
            for (int u = 0; u < W; ++u)
              if (r < T && base + r * W + u < n)
                f(k, base + r * W + u);
          }
      }
    }

    // Partials first, first + step, ... (Q of them) at once into s (those at or past T untouched).
    template <int T, int W, bool CONTIGUOUS, int Q, typename A, typename B>
    static __device__ __forceinline__ void partial_sums(int first, int step, int n, const A& a, const B& b,
                                                        float (&s)[Q]) {
      #pragma unroll
      for (int k = 0; k < Q; ++k)
        s[k] = 0.f;
      for_partials<T, W, CONTIGUOUS, Q>(first, step, n, [&](int k, int i) { s[k] = fmaf(a(i), b(i), s[k]); });
    }

    // partial_sums of two columns at once, a(i) b(i).x and a(i) b(i).y (b(i) a float2, e.g. a pair of halves read
    // together): each column's sums are partial_sums', element by element.
    template <int T, int W, bool CONTIGUOUS, int Q, typename A, typename B>
    static __device__ __forceinline__ void partial_sums2(int first, int step, int n, const A& a, const B& b,
                                                         float2 (&s)[Q]) {
      #pragma unroll
      for (int k = 0; k < Q; ++k)
        s[k] = make_float2(0.f, 0.f);
      for_partials<T, W, CONTIGUOUS, Q>(first, step, n, [&](int k, int i) {
        const float x = a(i);
        const float2 y = b(i);
        s[k].x = fmaf(x, y.x, s[k].x);
        s[k].y = fmaf(x, y.y, s[k].y);
      });
    }

  }
}

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

    // One output column x (of a block's 32) of `rows` (at most R) sums over n elements, a(k, i) b(k, i) for sum k,
    // each with the arithmetic of one thread's gemv: T partial sums (strided or contiguous, as above), each in order,
    // then combine's tree. The partials spread over the block's split_lanes y lanes (lane y sums partials y,
    // y + split_lanes, ...; with fewer partials than lanes, split_lanes / T lanes share a partial, each its share of
    // the rows: sum k where k % (split_lanes / T) is its turn), into sm (rows x T x 32 floats); then lane 0 combines
    // each sum and hands it to store(k, sum). Below `shared` the rows' b is one (b(0, i), read once for them all: a
    // clip's memory values, a beam search's prompt), from there each row's own. Every thread of the block calls it
    // (it synchronizes).
    template <int T, int W, bool CONTIGUOUS, int TREE, int R, typename A, typename B, typename S>
    static __device__ __forceinline__ void split_partials_rows(float* sm, int x, int y, int n, int rows, int shared,
                                                               const A& a, const B& b, const S& store) {
      constexpr bool pooled = T < split_lanes;
      constexpr int sharers = pooled ? split_lanes / T : 1, step = pooled ? T : split_lanes;
      const int turn = pooled ? y / T : 0;
      const auto mine = [&](int k) { return k < rows && k % sharers == turn; };
      for (int r = pooled ? y % T : y; r < T; r += step) {
        float s[R];
        #pragma unroll
        for (int k = 0; k < R; ++k)
          s[k] = 0.f;
        const auto add = [&](int i) {
          if (i < shared) {
            const float v = b(0, i);
            #pragma unroll
            for (int k = 0; k < R; ++k)
              if (mine(k))
                s[k] = fmaf(a(k, i), v, s[k]);
          } else {
            #pragma unroll
            for (int k = 0; k < R; ++k)
              if (mine(k))
                s[k] = fmaf(a(k, i), b(k, i), s[k]);
          }
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
          if (mine(k))
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

#pragma once

// Candidate arithmetics of cuBLAS's small products (one query against n keys, fp16, COMPUTE_32F), shared by the probes
// that recover them (ladder_cross_probe.cu, selfattn_recipe_probe.cu). Products of halves are exact in fp32, so a
// fused multiply-add is the product's sum.
#include <cuda_fp16.h>

__device__ __forceinline__ float f(__half h) { return __half2float(h); }

__device__ __forceinline__ unsigned pack(__half lo, __half hi) {
  return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
}

__device__ __forceinline__ void mma16816(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// sum over i < n of a[i * a_step] b[i] as T partial sums, thread r taking either the elements in vectors of w,
// (r + T j) w + u, or one contiguous chunk of ceil(n / T); the partials combined by a tree from the halves
// (tree 1: s_r += s_{r + T/2}, ... s_r += s_{r + 1}), from the neighbours (tree 2: s_r += s_{r + 1} for even r, then
// + 2, ...) or in order (tree 0).
__device__ __forceinline__ float reduce_partials(const __half* a, size_t a_step, const __half* b, int n, int T, int w,
                                                 bool contiguous, int tree) {
  float part[64];
  for (int r = 0; r < T; ++r) {
    float s = 0.f;
    if (contiguous) {
      const int chunk = (n + T - 1) / T;
      for (int i = r * chunk; i < min(n, (r + 1) * chunk); ++i) s = fmaf(f(a[i * a_step]), f(b[i]), s);
    } else {
      for (int base = r * w; base < n; base += T * w)
        for (int u = 0; u < w && base + u < n; ++u) s = fmaf(f(a[(base + u) * a_step]), f(b[base + u]), s);
    }
    part[r] = s;
  }
  if (tree == 1) {
    for (int off = T / 2; off > 0; off /= 2)
      for (int r = 0; r < off; ++r) part[r] += part[r + off];
    return part[0];
  }
  if (tree == 2) {
    for (int off = 1; off < T; off *= 2)
      for (int r = 0; r + off < T; r += 2 * off) part[r] += part[r + off];
    return part[0];
  }
  float sum = 0.f;
  for (int r = 0; r < T; ++r) sum += part[r];
  return sum;
}

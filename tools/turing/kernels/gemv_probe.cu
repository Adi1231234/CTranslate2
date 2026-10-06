// cuBLAS's arithmetic for one row of the decoder's products (primitives<CUDA>::gemm with m = 1, a group of one row:
// cuBLAS runs a gemv, internal::gemvx::kernel): y_j = sum_i W[j][i] x[i], fp16 in and out, COMPUTE_32F. A kernel that
// computes several such rows reading W once needs these bits (prof1: a ladder's single rows each re-read the
// weights, 17% of the ladders' GPU time). For each shape, every candidate below against cuBLAS over 3 fills:
//   thread t of T sums its elements, strided (vectors of w: (t + T j) w + u) or one contiguous chunk, into acc
//   accumulators (acc 2: u even and u odd apart, added at the end); the T partials then combined by a tree from the
//   halves (1), from the neighbours (2), in order (0), or per warp of 32 from the halves then the warps in order (3)
//   or by a tree from the halves (4).
// usage: gemv_probe -> per shape the candidates with 0 mismatches (or the closest)
#include <cstdio>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"

struct Recipe { int T, w, contiguous, acc, combine; };

__device__ float combine_partials(float* p, int T, int combine) {
  if (combine == 1) {
    for (int off = T / 2; off > 0; off /= 2) for (int r = 0; r < off; ++r) p[r] += p[r + off];
    return p[0];
  }
  if (combine == 2) {
    for (int off = 1; off < T; off *= 2) for (int r = 0; r + off < T; r += 2 * off) p[r] += p[r + off];
    return p[0];
  }
  if (combine == 0) {
    float s = 0.f;
    for (int r = 0; r < T; ++r) s += p[r];
    return s;
  }
  const int warps = T / 32;                                  // 3, 4: each warp from the halves, then the warps
  for (int wp = 0; wp < warps; ++wp) {
    float* q = p + 32 * wp;
    for (int off = 16; off > 0; off /= 2) for (int r = 0; r < off; ++r) q[r] += q[r + off];
  }
  if (combine == 3) {
    float s = 0.f;
    for (int wp = 0; wp < warps; ++wp) s += p[32 * wp];
    return s;
  }
  for (int off = warps / 2; off > 0; off /= 2) for (int wp = 0; wp < off; ++wp) p[32 * wp] += p[32 * (wp + off)];
  return p[0];
}

// One output a thread, the whole recipe serially (a probe, not a kernel to ship).
__global__ void candidate(const __half* W, const __half* x, __half* y, int n, int k, Recipe r) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  const __half* wj = W + (size_t)j * k;
  float part[512];
  for (int t = 0; t < r.T; ++t) {
    float a0 = 0.f, a1 = 0.f;
    const auto add = [&](int i, int u) {
      if (r.acc == 2 && (u & 1)) a1 = fmaf(__half2float(wj[i]), __half2float(x[i]), a1);
      else a0 = fmaf(__half2float(wj[i]), __half2float(x[i]), a0);
    };
    if (r.contiguous) {
      const int chunk = (k + r.T - 1) / r.T;
      for (int i = t * chunk, u = 0; i < min(k, (t + 1) * chunk); ++i, ++u) add(i, u);
    } else {
      for (int base = t * r.w; base < k; base += r.T * r.w)
        for (int u = 0; u < r.w && base + u < k; ++u) add(base + u, u);
    }
    part[t] = r.acc == 2 ? a0 + a1 : a0;
  }
  y[j] = __float2half_rn(combine_partials(part, r.T, r.combine));
}

int main() {
  cublasHandle_t h; CK(cublasCreate(&h));
  const int shapes[][2] = {{3840, 1280}, {1280, 1280}, {5120, 1280}, {1280, 5120}, {51872, 1280}, {51866, 1280}};
  std::vector<Recipe> recipes;
  for (const int T : {8, 16, 32, 64, 128, 256, 512})
    for (const int w : {1, 2, 4, 8})
      for (const int contiguous : {0, 1})
        for (const int acc : {1, 2})
          for (int combine = 0; combine <= 4; ++combine) {
            if (contiguous && w > 1) continue;               // a chunk has no vector width
            if (combine >= 3 && T < 64) continue;            // per warp needs two warps or more
            recipes.push_back({T, w, contiguous, acc, combine});
          }
  __half *W, *x, *y, *z; unsigned long long* dc;
  const size_t most = 51872ull * 5120;
  CK(cudaMalloc(&W, 2 * most)); CK(cudaMalloc(&x, 2 * 5120)); CK(cudaMalloc(&y, 2 * 51872)); CK(cudaMalloc(&z, 2 * 51872));
  CK(cudaMalloc(&dc, 8));
  const float one = 1.f, zero = 0.f;
  auto differ = [&](size_t count) {
    CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(y, z, count, dc);
    unsigned long long v; CK(cudaMemcpy(&v, dc, 8, cudaMemcpyDeviceToHost)); return v;
  };
  for (const auto& s : shapes) {
    const int n = s[0], k = s[1];
    std::vector<unsigned long long> miss(recipes.size(), 0);
    for (int fill_no = 0; fill_no < 3; ++fill_no) {
      fill<<<1024, 256>>>(W, (size_t)n * k, 21u + fill_no, -9, 0);
      fill<<<64, 256>>>(x, (size_t)k, 5u + fill_no, -4, 2);
      // primitives<CUDA>::gemm's call for one row: C (n x 1) = W^T-op x, column-major
      CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, k, &one, W, CUDA_R_16F, k, x, CUDA_R_16F, k, &zero, y,
                      CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      for (size_t c = 0; c < recipes.size(); ++c) {
        candidate<<<(n + 127) / 128, 128>>>(W, x, z, n, k, recipes[c]);
        miss[c] += differ(n);
      }
      CK(cudaGetLastError());
    }
    size_t best = 0;
    int exact = 0;
    for (size_t c = 0; c < recipes.size(); ++c) {
      if (miss[c] < miss[best]) best = c;
      if (miss[c] == 0) {
        const Recipe& r = recipes[c];
        printf("%5d x %4d: EXACT T %d w %d %s acc %d combine %d\n", n, k, r.T, r.w, r.contiguous ? "contig" : "strided",
               r.acc, r.combine);
        ++exact;
      }
    }
    if (!exact) {
      const Recipe& r = recipes[best];
      printf("%5d x %4d: none exact; closest T %d w %d %s acc %d combine %d: %llu of %d mismatched\n", n, k, r.T, r.w,
             r.contiguous ? "contig" : "strided", r.acc, r.combine, miss[best], 3 * n);
    }
  }
  return 0;
}

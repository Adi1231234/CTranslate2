// Which arithmetic does cuBLAS use for the Whisper decoder's second feed-forward product (C = A W^T, 1280 x 5120,
// fp16, COMPUTE_32F) at 1..16 rows, where hmma_probe.cu's candidates (split-K up to 8 slices) match nothing? Wider
// split-K candidates on the same mma.sync m16n8k16 chains: S = 2..16 slices of ceil(K / S) rounded up to G, the
// slices combined as
//   serial   out = half(acc0), then out = half(acc_s + out)          fp32 / halves  forward sum (fp32 / slices in half)
//   rfp32 / rhalves   the same summed last slice first                tfp32 / thalves  pairwise (tree) sum
// Prints, per row count, the candidates with no mismatch over 4 fills ("NONE" and the closest otherwise).
// usage: ffn2_probe [max M, default 16]
#include <algorithm>
#include <string>
#include <vector>
#include "probe_common.h"
#include "probe_data.cuh"

constexpr int N = 1280, K = 5120, MAXS = 16;

__device__ __forceinline__ unsigned pair_at(const __half* p, int rows, int r, size_t stride, int c) {
  return r < rows ? *reinterpret_cast<const unsigned*>(p + r * stride + c) : 0u;
}
__device__ __forceinline__ float rh(float x) { return __half2float(__float2half_rn(x)); }

// One warp per 16 x 8 tile of C.
__global__ void split_kernel(const __half* A, const __half* W, __half* C, int M, int S, int G, int mode) {
  const int g = threadIdx.x >> 2, t = threadIdx.x & 3, n0 = blockIdx.x * 8, m0 = blockIdx.y * 16;
  const int slice = ((K + S - 1) / S + G - 1) / G * G;
  float part[MAXS][4];
  for (int s = 0; s < S; ++s) {
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = s * slice; k < min(K, (s + 1) * slice); k += 16) {
      const unsigned b0 = pair_at(W, N, n0 + g, K, k + 2 * t), b1 = pair_at(W, N, n0 + g, K, k + 8 + 2 * t);
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(pair_at(A, M, m0 + g, K, k + 2 * t)), "r"(pair_at(A, M, m0 + g + 8, K, k + 2 * t)),
                     "r"(pair_at(A, M, m0 + g, K, k + 8 + 2 * t)), "r"(pair_at(A, M, m0 + g + 8, K, k + 8 + 2 * t)),
                     "r"(b0), "r"(b1));
    }
    for (int i = 0; i < 4; ++i) part[s][i] = (mode == 2 || mode == 4 || mode == 6) ? rh(d[i]) : d[i];
  }
  for (int i = 0; i < 4; ++i) {
    float out = 0.f;
    if (mode == 0) {
      for (int s = 0; s < S; ++s) out = rh(s == 0 ? part[s][i] : part[s][i] + out);
    } else if (mode == 1 || mode == 2) {
      out = part[0][i];
      for (int s = 1; s < S; ++s) out += part[s][i];
    } else if (mode == 3 || mode == 4) {
      out = part[S - 1][i];
      for (int s = S - 2; s >= 0; --s) out += part[s][i];
    } else {
      float v[MAXS];
      int n = S;
      for (int s = 0; s < S; ++s) v[s] = part[s][i];
      while (n > 1) {
        const int h = (n + 1) / 2;
        for (int s = 0; s + h < n; ++s) v[s] += v[s + h];
        n = h;
      }
      out = v[0];
    }
    const int row = m0 + g + 8 * (i / 2), col = n0 + 2 * t + i % 2;
    if (row < M && col < N) C[(size_t)row * N + col] = __float2half_rn(out);
  }
}

int main(int argc, char** argv) {
  const int max_m = argc > 1 ? atoi(argv[1]) : 16;
  struct Cand { std::string name; int s, g, mode; };
  const char* modes[7] = {"serial", "fp32", "halves", "rfp32", "rhalves", "tfp32", "thalves"};
  std::vector<Cand> cands;
  for (int s = 2; s <= MAXS; ++s)
    for (int g : {16, 32, 64, 128, 256, 512, 640, 1024})
      for (int mode = 0; mode < 7; ++mode)
        cands.push_back({std::string(modes[mode]) + " " + std::to_string(s) + "/" + std::to_string(g), s, g, mode});
  cublasHandle_t h; CK(cublasCreate(&h));
  __half *A, *W, *C, *R; unsigned long long* dc;
  CK(cudaMalloc(&A, 2ull * 64 * K)); CK(cudaMalloc(&W, 2ull * N * K));
  CK(cudaMalloc(&C, 2ull * 64 * N)); CK(cudaMalloc(&R, 2ull * 64 * N)); CK(cudaMalloc(&dc, 8));
  for (int M = 1; M <= max_m; ++M) {
    std::vector<unsigned long long> bad(cands.size(), 0);
    for (int f = 0; f < 4; ++f) {
      fill<<<256, 256>>>(A, (size_t)M * K, 17u * f + M, -9 + 2 * f, 1 + f);
      fill<<<1024, 256>>>(W, (size_t)N * K, 101u * f + N + K, -15 + f, -3 + 2 * f);
      const float alpha = 1.f, beta = 0.f;
      CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K,
                      &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      for (size_t c = 0; c < cands.size(); ++c) {
        if (bad[c] > 0 && f > 0) continue;                  // already out
        split_kernel<<<dim3(N / 8, (M + 15) / 16), 32>>>(A, W, R, M, cands[c].s, cands[c].g, cands[c].mode);
        CK(cudaGetLastError());
        CK(cudaMemset(dc, 0, 8)); count_diff<<<256, 256>>>(C, R, (size_t)M * N, dc);
        unsigned long long d; CK(cudaMemcpy(&d, dc, 8, cudaMemcpyDeviceToHost)); bad[c] += d;
      }
    }
    std::string hits;
    for (size_t c = 0; c < cands.size(); ++c)
      if (!bad[c]) hits += " [" + cands[c].name + "]";
    const size_t best = std::min_element(bad.begin(), bad.end()) - bad.begin();
    printf("M %2d: %s\n", M, hits.empty() ? ("NONE, closest " + cands[best].name + " (" +
                                             std::to_string(bad[best]) + ")").c_str() : hits.c_str());
    fflush(stdout);
  }
  return 0;
}
